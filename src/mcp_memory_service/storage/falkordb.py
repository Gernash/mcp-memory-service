"""
FalkorDB + LanceDB storage backend for MCP Memory Service.

FalkorDB (NAS): Memory node storage, Tag relationships, metadata.
LanceDB (local): Vector/semantic search — per-spoke isolated table.

Architecture decisions: ADR-HUB-119, docs/research/workspace-V4/MCP-MEMORY-ARCHITECTURE-2026-03-18.md

Config env vars:
    MCP_MEMORY_SPOKE_ID        Spoke identity → graph/table name memory_{spoke_id} (default: hub)
    FALKORDB_HOST              FalkorDB host   (default: 192.168.1.100)
    FALKORDB_PORT              FalkorDB port   (default: 6379)
    FALKORDB_PASSWORD          FalkorDB auth   (required)
    MCP_MEMORY_LANCEDB_PATH    LanceDB directory (default: ~/.mcp_memory/lancedb)
    MCP_EMBEDDING_MODEL        Embedding model   (default: all-MiniLM-L6-v2)
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import pyarrow as pa

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult

logger = logging.getLogger(__name__)

# Module-level embedding model cache — avoid reloading across instances
_MODEL_CACHE: Dict[str, Any] = {}


class FalkorDBMemoryStorage(MemoryStorage):
    """
    FalkorDB + LanceDB memory storage backend.

    FalkorDB stores Memory nodes and Tag relationships.
    LanceDB stores embeddings for vector/semantic search.

    Graph schema:
        (:Memory {content_hash, content, memory_type, tags_json,
                  metadata_json, created_at, created_at_iso,
                  updated_at, updated_at_iso})
        -[:HAS_TAG]->
        (:Tag {name})
    """

    # ------------------------------------------------------------------ properties

    @property
    def max_content_length(self) -> Optional[int]:
        return None  # No limit imposed by this backend

    @property
    def supports_chunking(self) -> bool:
        return False  # Memory entries are atomic

    # ------------------------------------------------------------------ init

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        spoke_id = os.getenv("MCP_MEMORY_SPOKE_ID", "hub")
        self.graph_name = f"memory_{spoke_id}"
        self.lancedb_table_name = f"memory_{spoke_id}"
        self.lancedb_path = os.getenv(
            "MCP_MEMORY_LANCEDB_PATH",
            os.path.join(os.path.expanduser("~"), ".mcp_memory", "lancedb"),
        )
        self.falkordb_host = os.getenv("FALKORDB_HOST", "192.168.1.100")
        self.falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
        self.falkordb_username = os.getenv("FALKORDB_USERNAME", "")
        self.falkordb_password = os.getenv("FALKORDB_PASSWORD", "3Zy!2!M7jF99vgRp")
        self.embedding_model_name = embedding_model

        self._graph = None          # FalkorDB Graph object
        self._lance_db = None       # LanceDB connection
        self._lance_table = None    # LanceDB table
        self._embedding_model = None
        self._embedding_dim = 384   # Updated on init once model loads
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="falkordb"
        )
        self._initialized = False

        logger.info(
            f"FalkorDBMemoryStorage created — graph: {self.graph_name}, "
            f"lancedb: {self.lancedb_path}/{self.lancedb_table_name}"
        )

    # ------------------------------------------------------------------ internal helpers

    async def _query(self, cypher: str, params: Optional[Dict] = None):
        """Execute a Cypher query asynchronously (FalkorDB client is sync)."""
        graph = self._graph
        p = params or {}
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: graph.query(cypher, p)
        )

    def _load_embedding_model(self) -> None:
        """Load SentenceTransformer — cached at module level."""
        global _MODEL_CACHE
        if self.embedding_model_name in _MODEL_CACHE:
            self._embedding_model = _MODEL_CACHE[self.embedding_model_name]
        else:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model_name)
                _MODEL_CACHE[self.embedding_model_name] = model
                self._embedding_model = model
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.error(
                    "sentence_transformers not installed — vector search unavailable"
                )
                return

        # Detect actual dimension
        try:
            sample = self._embedding_model.encode(["dim-check"], convert_to_numpy=True)
            self._embedding_dim = int(sample.shape[1])
        except Exception:
            self._embedding_dim = 384

    def _embed(self, text: str) -> List[float]:
        """Generate embedding vector. Returns zero vector on failure."""
        if self._embedding_model is None:
            return [0.0] * self._embedding_dim
        try:
            vec = self._embedding_model.encode([text], convert_to_numpy=True)
            return vec[0].tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return [0.0] * self._embedding_dim

    def _row_to_memory(self, row: list) -> Optional[Memory]:
        """Convert a FalkorDB result row (node at index 0) to Memory."""
        try:
            props = row[0].properties
            return Memory(
                content=props["content"],
                content_hash=props["content_hash"],
                tags=json.loads(props.get("tags_json", "[]")),
                memory_type=props.get("memory_type") or None,
                metadata=json.loads(props.get("metadata_json", "{}")),
                created_at=props.get("created_at"),
                created_at_iso=props.get("created_at_iso"),
                updated_at=props.get("updated_at"),
                updated_at_iso=props.get("updated_at_iso"),
            )
        except Exception as e:
            logger.error(f"Failed to convert FalkorDB row to Memory: {e}")
            return None

    # ------------------------------------------------------------------ initialize

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            import falkordb as fdb

            # Connect to FalkorDB (sync — run in executor)
            def _connect():
                kwargs = dict(
                    host=self.falkordb_host,
                    port=self.falkordb_port,
                    password=self.falkordb_password,
                )
                if self.falkordb_username:
                    kwargs["username"] = self.falkordb_username
                db = fdb.FalkorDB(**kwargs)
                return db.select_graph(self.graph_name)

            loop = asyncio.get_event_loop()
            self._graph = await loop.run_in_executor(self._executor, _connect)

            # Create index on content_hash (idempotent — ignore if exists)
            try:
                await self._query(
                    "CREATE INDEX FOR (m:Memory) ON (m.content_hash)"
                )
            except Exception:
                pass  # Index already exists

            # Load embedding model
            self._load_embedding_model()

            # Connect to LanceDB and create/open memory table
            os.makedirs(self.lancedb_path, exist_ok=True)
            self._lance_db = lancedb.connect(self.lancedb_path)

            schema = pa.schema([
                pa.field("content_hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._embedding_dim)),
            ])

            if self.lancedb_table_name in self._lance_db.table_names():
                self._lance_table = self._lance_db.open_table(self.lancedb_table_name)
            else:
                self._lance_table = self._lance_db.create_table(
                    self.lancedb_table_name, schema=schema
                )

            self._initialized = True
            logger.info(
                f"FalkorDB storage initialized — graph: {self.graph_name}, "
                f"embedding_dim: {self._embedding_dim}"
            )

        except Exception as e:
            logger.error(f"FalkorDB storage initialization failed: {e}")
            raise

    # ------------------------------------------------------------------ store

    async def store(
        self, memory: Memory, skip_semantic_dedup: bool = False
    ) -> Tuple[bool, str]:
        try:
            # Exact-hash dedup check
            exists = await self._query(
                "MATCH (m:Memory {content_hash: $hash}) RETURN m LIMIT 1",
                {"hash": memory.content_hash},
            )
            if exists.result_set:
                return False, "Duplicate content detected (exact match)"

            tags_json = json.dumps(memory.tags or [])
            metadata_json = json.dumps(memory.metadata or {})
            now = time.time()

            # Write Memory node to FalkorDB
            await self._query(
                """
                CREATE (m:Memory {
                    content_hash:  $hash,
                    content:       $content,
                    memory_type:   $memory_type,
                    tags_json:     $tags_json,
                    metadata_json: $metadata_json,
                    created_at:    $created_at,
                    created_at_iso: $created_at_iso,
                    updated_at:    $updated_at,
                    updated_at_iso: $updated_at_iso
                })
                """,
                {
                    "hash": memory.content_hash,
                    "content": memory.content,
                    "memory_type": memory.memory_type or "",
                    "tags_json": tags_json,
                    "metadata_json": metadata_json,
                    "created_at": memory.created_at or now,
                    "created_at_iso": memory.created_at_iso or "",
                    "updated_at": memory.updated_at or now,
                    "updated_at_iso": memory.updated_at_iso or "",
                },
            )

            # Create Tag nodes and HAS_TAG relationships
            for tag in (memory.tags or []):
                await self._query(
                    """
                    MERGE (t:Tag {name: $tag})
                    WITH t
                    MATCH (m:Memory {content_hash: $hash})
                    MERGE (m)-[:HAS_TAG]->(t)
                    """,
                    {"tag": tag, "hash": memory.content_hash},
                )

            # Write embedding to LanceDB
            # If LanceDB fails, roll back the FalkorDB node to avoid split-brain
            try:
                vector = self._embed(memory.content)
                self._lance_table.add([{
                    "content_hash": memory.content_hash,
                    "vector": vector,
                }])
            except Exception as lance_err:
                logger.error(
                    f"LanceDB write failed — rolling back FalkorDB node "
                    f"{memory.content_hash}: {lance_err}"
                )
                await self._query(
                    "MATCH (m:Memory {content_hash: $hash}) DETACH DELETE m",
                    {"hash": memory.content_hash},
                )
                return False, f"Store failed (LanceDB rollback): {lance_err}"

            logger.info(f"Stored memory: {memory.content_hash}")
            return True, "Memory stored successfully"

        except Exception as e:
            logger.error(f"store() failed: {e}")
            return False, f"Store failed: {e}"

    # ------------------------------------------------------------------ retrieve (vector search)

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[MemoryQueryResult]:
        try:
            if self._lance_table is None:
                return []

            vector = self._embed(query)

            # Over-fetch from LanceDB to leave room for tag filtering
            fetch_limit = n_results * 4 if tags else n_results
            lance_rows = (
                self._lance_table
                .search(vector)
                .limit(fetch_limit)
                .to_list()
            )

            results: List[MemoryQueryResult] = []
            seen: set = set()

            for row in lance_rows:
                if len(results) >= n_results:
                    break
                content_hash = row["content_hash"]
                if content_hash in seen:
                    continue
                seen.add(content_hash)

                # Optional tag pre-filter
                if tags:
                    tag_check = await self._query(
                        """
                        MATCH (m:Memory {content_hash: $hash})-[:HAS_TAG]->(t:Tag)
                        WHERE t.name IN $tags
                        RETURN m LIMIT 1
                        """,
                        {"hash": content_hash, "tags": tags},
                    )
                    if not tag_check.result_set:
                        continue

                mem_result = await self._query(
                    "MATCH (m:Memory {content_hash: $hash}) RETURN m LIMIT 1",
                    {"hash": content_hash},
                )
                if not mem_result.result_set:
                    continue

                memory = self._row_to_memory(mem_result.result_set[0])
                if memory:
                    distance = float(row.get("_distance", 0.0))
                    score = max(0.0, 1.0 - distance)
                    if score < min_confidence:
                        continue
                    results.append(MemoryQueryResult(memory=memory, relevance_score=score))

            return results

        except Exception as e:
            logger.error(f"retrieve() failed: {e}")
            return []

    # ------------------------------------------------------------------ tag search

    async def search_by_tag(
        self,
        tags: List[str],
        time_start: Optional[float] = None,
    ) -> List[Memory]:
        try:
            cypher = """
                MATCH (m:Memory)-[:HAS_TAG]->(t:Tag)
                WHERE t.name IN $tags
            """
            params: Dict[str, Any] = {"tags": tags}
            if time_start is not None:
                cypher += " AND m.created_at >= $time_start"
                params["time_start"] = time_start
            cypher += " RETURN DISTINCT m ORDER BY m.created_at DESC"

            result = await self._query(cypher, params)
            return [m for m in (self._row_to_memory(r) for r in result.result_set) if m]

        except Exception as e:
            logger.error(f"search_by_tag() failed: {e}")
            return []

    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[Memory]:
        try:
            params: Dict[str, Any] = {"tags": tags}

            if operation.upper() == "AND":
                # Memory must match ALL tags — use count approach (broadly compatible Cypher)
                params["tag_count"] = len(tags)
                cypher = """
                    MATCH (m:Memory)-[:HAS_TAG]->(t:Tag)
                    WHERE t.name IN $tags
                    WITH m, count(DISTINCT t) AS matched
                    WHERE matched = $tag_count
                """
            else:
                # ANY tag match
                cypher = """
                    MATCH (m:Memory)-[:HAS_TAG]->(t:Tag)
                    WHERE t.name IN $tags
                    WITH m
                """

            if time_start is not None:
                cypher += " AND m.created_at >= $time_start"
                params["time_start"] = time_start
                if time_end is not None:
                    cypher += " AND m.created_at <= $time_end"
                    params["time_end"] = time_end
            elif time_end is not None:
                cypher += " AND m.created_at <= $time_end"
                params["time_end"] = time_end

            cypher += " RETURN DISTINCT m ORDER BY m.created_at DESC"

            result = await self._query(cypher, params)
            return [m for m in (self._row_to_memory(r) for r in result.result_set) if m]

        except Exception as e:
            logger.error(f"search_by_tags() failed: {e}")
            return []

    # ------------------------------------------------------------------ delete

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        try:
            # FalkorDB: DETACH DELETE removes the node and all its relationships
            await self._query(
                "MATCH (m:Memory {content_hash: $hash}) DETACH DELETE m",
                {"hash": content_hash},
            )

            # LanceDB: remove the vector row
            try:
                self._lance_table.delete(f"content_hash = '{content_hash}'")
            except Exception as e:
                logger.warning(f"LanceDB delete partial failure for {content_hash}: {e}")

            return True, "Memory deleted"

        except Exception as e:
            logger.error(f"delete() failed: {e}")
            return False, f"Delete failed: {e}"

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        try:
            result = await self._query(
                """
                MATCH (m:Memory)-[:HAS_TAG]->(:Tag {name: $tag})
                RETURN m.content_hash
                """,
                {"tag": tag},
            )
            hashes = [row[0] for row in result.result_set]
            count = 0
            for h in hashes:
                success, _ = await self.delete(h)
                if success:
                    count += 1
            return count, f"Deleted {count} memories with tag '{tag}'"

        except Exception as e:
            logger.error(f"delete_by_tag() failed: {e}")
            return 0, f"delete_by_tag failed: {e}"

    # ------------------------------------------------------------------ lookup

    async def get_by_exact_content(self, content: str) -> List[Memory]:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        memory = await self.get_by_hash(content_hash)
        return [memory] if memory else []

    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        try:
            result = await self._query(
                "MATCH (m:Memory {content_hash: $hash}) RETURN m LIMIT 1",
                {"hash": content_hash},
            )
            if not result.result_set:
                return None
            return self._row_to_memory(result.result_set[0])

        except Exception as e:
            logger.error(f"get_by_hash() failed: {e}")
            return None

    # ------------------------------------------------------------------ maintenance

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        # store() uses content_hash uniqueness check — duplicates cannot occur
        # under normal operation. No action needed.
        return 0, "No duplicates (FalkorDB content_hash uniqueness enforced on store)"

    async def update_memory_metadata(
        self,
        content_hash: str,
        updates: Dict[str, Any],
        preserve_timestamps: bool = True,
    ) -> Tuple[bool, str]:
        try:
            memory = await self.get_by_hash(content_hash)
            if not memory:
                return False, f"Memory not found: {content_hash}"

            if "tags" in updates:
                memory.tags = updates["tags"]
            if "metadata" in updates:
                memory.metadata.update(updates["metadata"])
            if "memory_type" in updates:
                memory.memory_type = updates["memory_type"]

            now = time.time()
            updated_at_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

            await self._query(
                """
                MATCH (m:Memory {content_hash: $hash})
                SET m.tags_json     = $tags_json,
                    m.metadata_json = $metadata_json,
                    m.memory_type   = $memory_type,
                    m.updated_at    = $updated_at,
                    m.updated_at_iso = $updated_at_iso
                """,
                {
                    "hash": content_hash,
                    "tags_json": json.dumps(memory.tags),
                    "metadata_json": json.dumps(memory.metadata),
                    "memory_type": memory.memory_type or "",
                    "updated_at": now,
                    "updated_at_iso": updated_at_iso,
                },
            )

            # Rebuild HAS_TAG edges
            await self._query(
                "MATCH (m:Memory {content_hash: $hash})-[r:HAS_TAG]->() DELETE r",
                {"hash": content_hash},
            )
            for tag in (memory.tags or []):
                await self._query(
                    """
                    MERGE (t:Tag {name: $tag})
                    WITH t
                    MATCH (m:Memory {content_hash: $hash})
                    MERGE (m)-[:HAS_TAG]->(t)
                    """,
                    {"tag": tag, "hash": content_hash},
                )

            return True, "Memory metadata updated"

        except Exception as e:
            logger.error(f"update_memory_metadata() failed: {e}")
            return False, f"Update failed: {e}"

    # ------------------------------------------------------------------ optional overrides

    async def get_stats(self) -> Dict[str, Any]:
        try:
            result = await self._query("MATCH (m:Memory) RETURN count(m)")
            count = result.result_set[0][0] if result.result_set else 0
            return {
                "total_memories": count,
                "storage_backend": "falkordb",
                "graph_name": self.graph_name,
                "lancedb_table": self.lancedb_table_name,
                "lancedb_path": self.lancedb_path,
                "embedding_model": self.embedding_model_name,
                "embedding_dim": self._embedding_dim,
                "status": "operational",
            }
        except Exception as e:
            return {"storage_backend": "falkordb", "status": f"error: {e}"}

    async def get_all_tags(self) -> List[str]:
        try:
            result = await self._query(
                "MATCH (t:Tag) RETURN t.name ORDER BY t.name"
            )
            return [row[0] for row in result.result_set]
        except Exception as e:
            logger.error(f"get_all_tags() failed: {e}")
            return []

    async def get_all_memories(
        self,
        limit: int = None,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Memory]:
        try:
            params: Dict[str, Any] = {}
            conditions: List[str] = []

            if tags:
                cypher = "MATCH (m:Memory)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags"
                params["tags"] = tags
            else:
                cypher = "MATCH (m:Memory)"

            if memory_type:
                conditions.append("m.memory_type = $memory_type")
                params["memory_type"] = memory_type

            if conditions:
                cypher += (" AND " if tags else " WHERE ") + " AND ".join(conditions)

            cypher += " RETURN DISTINCT m ORDER BY m.created_at DESC"

            if offset:
                cypher += f" SKIP {int(offset)}"
            if limit:
                cypher += f" LIMIT {int(limit)}"

            result = await self._query(cypher, params)
            return [m for m in (self._row_to_memory(r) for r in result.result_set) if m]

        except Exception as e:
            logger.error(f"get_all_memories() failed: {e}")
            return []

    # ------------------------------------------------------------------ conflict resolution (v10.30.0)

    async def get_conflicts(self) -> List[Dict[str, Any]]:
        """Return unresolved conflict pairs — memories connected by :CONTRADICTS edges
        where neither node has been superseded."""
        try:
            result = await self._query(
                """
                MATCH (m1:Memory)-[c:CONTRADICTS]->(m2:Memory)
                WHERE (m1.superseded_by IS NULL OR m1.superseded_by = '')
                  AND (m2.superseded_by IS NULL OR m2.superseded_by = '')
                  AND m1.content_hash < m2.content_hash
                RETURN m1.content_hash AS hash_a,
                       m2.content_hash AS hash_b,
                       m1.content      AS content_a,
                       m2.content      AS content_b,
                       c.similarity    AS similarity,
                       c.divergence    AS divergence,
                       c.detected_at   AS detected_at
                """,
                {},
            )
            conflicts = []
            for row in result.result_set:
                conflicts.append({
                    "hash_a":      row[0],
                    "hash_b":      row[1],
                    "content_a":   row[2],
                    "content_b":   row[3],
                    "similarity":  row[4],
                    "divergence":  row[5],
                    "detected_at": row[6],
                })
            return conflicts
        except Exception as e:
            logger.error(f"get_conflicts() failed: {e}")
            return []

    async def resolve_conflict(self, winner_hash: str, loser_hash: str) -> Tuple[bool, str]:
        """Resolve a conflict: mark loser as superseded, boost winner confidence."""
        import time as _time
        now = _time.time()
        try:
            # Verify both exist and are active
            for h, label in ((winner_hash, "Winner"), (loser_hash, "Loser")):
                check = await self._query(
                    """MATCH (m:Memory {content_hash: $h})
                       WHERE (m.superseded_by IS NULL OR m.superseded_by = '')
                       RETURN m LIMIT 1""",
                    {"h": h},
                )
                if not check.result_set:
                    return False, f"{label} memory {h[:8]} not found or already superseded"

            # Mark loser as superseded
            await self._query(
                "MATCH (m:Memory {content_hash: $h}) SET m.superseded_by = $winner",
                {"h": loser_hash, "winner": winner_hash},
            )

            # Boost winner: confidence = 1.0, last_accessed = now
            await self._query(
                "MATCH (m:Memory {content_hash: $h}) SET m.confidence = 1.0, m.last_accessed = $now",
                {"h": winner_hash, "now": int(now)},
            )

            # Remove the CONTRADICTS edge(s) between these two
            await self._query(
                """MATCH (m1:Memory {content_hash: $a})-[c:CONTRADICTS]-(m2:Memory {content_hash: $b})
                   DELETE c""",
                {"a": winner_hash, "b": loser_hash},
            )

            logger.info(f"resolve_conflict: {winner_hash[:8]} wins over {loser_hash[:8]}")
            return True, f"Conflict resolved: {winner_hash[:8]} supersedes {loser_hash[:8]}"
        except Exception as e:
            logger.error(f"resolve_conflict() failed: {e}")
            return False, str(e)
