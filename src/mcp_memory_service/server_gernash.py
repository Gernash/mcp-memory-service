"""
Gernash Memory Server — namespace-isolated MCP server over mcp_memory_service + FalkorDB.

Exposes memory tools with automatic namespace isolation via SPOKE_ID:
- Every stored memory is tagged with spoke:<SPOKE_ID>
- Every search is filtered to spoke:<SPOKE_ID> results only
- Multiple workspaces share one FalkorDB instance with zero cross-contamination

Transport: stdio.

Environment variables:
    SPOKE_ID              Namespace identifier (default: "hub")
    MCP_MEMORY_STORAGE_BACKEND  Storage backend (default: "falkordb")
    FALKORDB_HOST         FalkorDB host (default: "localhost")
    FALKORDB_PORT         FalkorDB port (default: "6379")
    FALKORDB_PASSWORD     FalkorDB password (optional)
    MEMORY_LANCEDB_PATH   Path for LanceDB hub_facts store
                          (default: ~/.gernash/memory/localdb)
    LM_STUDIO_BASE_URL    Embedding server base URL
                          (default: http://localhost:1234/v1)
    ENABLE_MEMORY_DELETE  Set to "true" to enable memory_delete tool
    WAKEUP_LOOKBACK_DAYS  Days of history for memory_wakeup (default: 30)

Registration in mcp.json:
    "gernash-memory": {
        "type": "stdio",
        "command": "/path/to/.venv/Scripts/gernash-memory-server",
        "env": { "SPOKE_ID": "myproject" }
    }
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("MCP_MEMORY_STORAGE_BACKEND", "falkordb")

from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mcp_memory_service.storage.factory import create_storage_instance
from mcp_memory_service.models.memory import Memory

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SPOKE_ID: str = os.environ.get("SPOKE_ID", "hub").strip().lower()
_SPOKE_TAG: str = f"spoke:{SPOKE_ID}"

# LanceDB path — no hard-coded workspace paths
_LANCEDB_PATH: str = os.environ.get(
    "MEMORY_LANCEDB_PATH",
    str(Path.home() / ".gernash" / "memory" / "localdb"),
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_WAKEUP_LOOKBACK_DAYS = 30
MAX_SEARCH_LIMIT = 50
WAKEUP_SECTION_LIMIT = 5
WAKEUP_FALLBACK_LIMIT = 10

# --------------------------------------------------------------------------
# Storage singleton
# --------------------------------------------------------------------------

_storage = None
_storage_lock = asyncio.Lock()


async def _get_storage():
    global _storage
    async with _storage_lock:
        if _storage is None:
            try:
                _storage = await create_storage_instance(server_type="mcp")
            except Exception as exc:
                logger.exception("Failed to initialize memory storage")
                raise RuntimeError("Failed to initialize memory storage") from exc
    return _storage


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(server):
    """Pre-warm storage before first tool call to avoid MCP timeouts."""
    await _get_storage()
    logger.info("gernash-memory: storage pre-warmed (spoke=%s)", SPOKE_ID)
    yield


mcp = FastMCP(
    name=f"gernash-memory-{SPOKE_ID}",
    lifespan=_lifespan,
    instructions=(
        f"Namespace-isolated memory store for '{SPOKE_ID}'. "
        "Use memory_store to persist important context, decisions, and findings. "
        "Use memory_search to retrieve relevant past context by semantic query. "
        "All memories are automatically scoped to this namespace — no cross-contamination."
    ),
)


# --------------------------------------------------------------------------
# Response Models
# --------------------------------------------------------------------------

class MemoryStoreResult(BaseModel):
    success: bool
    message: str
    spoke: str


class MemoryItem(BaseModel):
    content_hash: str
    content: str
    tags: list[str]
    score: Optional[float]
    memory_type: str
    created_at: str
    spoke: str


class MemorySearchResult(BaseModel):
    memories: list[MemoryItem]
    spoke: str
    error: Optional[str] = None


class WakeupMemoryItem(BaseModel):
    content: str
    score: Optional[float]
    created_at: Optional[str] = None


class MemoryWakeupResult(BaseModel):
    spoke: str
    active_tasks: list[WakeupMemoryItem]
    recent_decisions: list[WakeupMemoryItem]
    blockers: list[WakeupMemoryItem]
    error: Optional[str] = None


class SessionCloseResult(BaseModel):
    success: bool
    message: str
    spoke: str
    session_metadata: dict = Field(default_factory=dict)


class MemoryDeleteResult(BaseModel):
    success: bool
    message: str
    spoke: str
    content_hash: str
    dry_run: bool = False
    deleted: bool = False
    error: Optional[str] = None


class FactResult(BaseModel):
    fact_text: str
    confidence: float
    session: str
    supersedes: Optional[str] = None


class FactRecallResult(BaseModel):
    facts: list[FactResult]
    spoke: str
    query: str
    include_superseded: bool = False
    error: Optional[str] = None


class FactStoreResult(BaseModel):
    success: bool
    fact_id: str
    message: str
    spoke: str
    skipped: bool = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _inject_spoke_tag(tags: list[str]) -> list[str]:
    merged: list[str] = []
    for tag in [_SPOKE_TAG, *tags]:
        if tag and tag not in merged:
            merged.append(tag)
    return merged


def _parse_tags(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [tag for tag in re.split(r"[\s,]+", raw.strip()) if tag]
    return []


def _memory_delete_enabled() -> bool:
    return os.environ.get("ENABLE_MEMORY_DELETE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_iso_timestamp(memory: Memory) -> str:
    if memory.created_at_iso:
        return memory.created_at_iso
    if memory.created_at is not None:
        return datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat()
    return ""


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_SEARCH_LIMIT))


def _wakeup_lookback_days() -> int:
    raw = os.environ.get("WAKEUP_LOOKBACK_DAYS", str(DEFAULT_WAKEUP_LOOKBACK_DAYS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_WAKEUP_LOOKBACK_DAYS
    return value if value > 0 else DEFAULT_WAKEUP_LOOKBACK_DAYS


def _memory_sort_key(memory: Memory) -> float:
    return memory.created_at or 0.0


def _is_recent_memory(memory: Memory, time_cutoff: float) -> bool:
    return (memory.created_at or 0.0) >= time_cutoff


def _is_session_close_memory(memory: Memory) -> bool:
    return "session-close" in set(memory.tags or [])


def _format_wakeup_memories(memories: list[Memory]) -> list[WakeupMemoryItem]:
    return [
        WakeupMemoryItem(
            content=memory.content[:400],
            score=None,
            created_at=_normalize_iso_timestamp(memory) or None,
        )
        for memory in memories[:WAKEUP_SECTION_LIMIT]
    ]


def _categorize_fallback_memory(memory: Memory) -> str:
    tags = set(memory.tags or [])
    if "blocker" in tags:
        return "blockers"
    if "decision" in tags:
        return "recent_decisions"
    return "active_tasks"


def _dedupe_memories(memories: list[Memory]) -> list[Memory]:
    deduped: list[Memory] = []
    seen: set[str] = set()
    for memory in memories:
        identity = memory.content_hash or f"{memory.content}|{memory.created_at or 0.0}"
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(memory)
    return deduped


# --------------------------------------------------------------------------
# Hub facts recaller
# --------------------------------------------------------------------------

_HUB_FACTS_TABLE = "hub_facts_embeddings"
_HUB_FACTS_GRAPH = "hub_facts"
_HUB_FACTS_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
_HUB_FACTS_LIMIT = 20
_HUB_FACTS_DIM = 768


class _HubFactsRecaller:
    _instance: Optional[_HubFactsRecaller] = None

    def __init__(self) -> None:
        self._db = None
        self._table = None
        self._client = None
        self._graph = None

    @classmethod
    def get(cls) -> _HubFactsRecaller:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_ready(self) -> None:
        if self._db is None:
            import lancedb as _lancedb
            self._db = _lancedb.connect(_LANCEDB_PATH)
        if self._table is None:
            existing = self._db.list_tables()
            if _HUB_FACTS_TABLE not in existing.tables:
                raise RuntimeError(
                    f"Hub facts table '{_HUB_FACTS_TABLE}' not found — "
                    "run chat_ingest pipeline first"
                )
            self._table = self._db.open_table(_HUB_FACTS_TABLE)
        if self._client is None:
            from openai import OpenAI
            base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
            self._client = OpenAI(base_url=base_url, api_key="not-needed")
        if self._graph is None:
            import falkordb as _falkordb
            host = os.environ.get("FALKORDB_HOST", "localhost")
            port = int(os.environ.get("FALKORDB_PORT", "6379"))
            password = os.environ.get("FALKORDB_PASSWORD", "")
            conn_kwargs: dict = {"host": host, "port": port}
            if password:
                conn_kwargs["password"] = password
            self._graph = _falkordb.FalkorDB(**conn_kwargs).select_graph(_HUB_FACTS_GRAPH)

    def query(self, query: str, include_superseded: bool = False, limit: int = _HUB_FACTS_LIMIT) -> list[FactResult]:
        self._ensure_ready()
        resp = self._client.embeddings.create(model=_HUB_FACTS_EMBED_MODEL, input=[query])
        vector = resp.data[0].embedding
        rows = self._table.search(vector).limit(limit * 3).to_list()
        candidates: list[tuple[str, float]] = []
        seen: set[str] = set()
        for row in rows:
            if not include_superseded and row.get("superseded", False):
                continue
            fid = row["fact_id"]
            if fid not in seen:
                seen.add(fid)
                candidates.append((fid, float(row.get("_distance", 1.0))))
            if len(candidates) >= limit:
                break
        if not candidates:
            return []
        results: list[FactResult] = []
        for fid, dist in candidates:
            try:
                if include_superseded:
                    cypher = "MATCH (f:Fact {content_hash: $h}) RETURN f.content, f.session_id, f.superseded_by LIMIT 1"
                else:
                    cypher = "MATCH (f:Fact {content_hash: $h}) WHERE (f.superseded_by IS NULL OR f.superseded_by = '') RETURN f.content, f.session_id, f.superseded_by LIMIT 1"
                r = self._graph.query(cypher, {"h": fid})
                if r.result_set:
                    row_data = r.result_set[0]
                    superseded_by = row_data[2]
                    results.append(FactResult(
                        fact_text=row_data[0] or "",
                        confidence=round(max(0.0, 1.0 - dist), 4),
                        session=row_data[1] or "",
                        supersedes=superseded_by if superseded_by else None,
                    ))
            except Exception as exc:
                logger.warning("memory_recall FalkorDB lookup failed for %s: %s", fid[:8], exc)
        return results


# --------------------------------------------------------------------------
# Hub facts write helper
# --------------------------------------------------------------------------

def _write_hub_fact(content: str, fact_id: str, session_id: str, now: float, now_iso: str) -> dict:
    import lancedb as _lancedb
    import falkordb as _falkordb
    from openai import OpenAI

    base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    resp = client.embeddings.create(model=_HUB_FACTS_EMBED_MODEL, input=[content])
    vector = resp.data[0].embedding

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))
    password = os.environ.get("FALKORDB_PASSWORD", "")
    conn_kwargs: dict = {"host": host, "port": port}
    if password:
        conn_kwargs["password"] = password
    graph = _falkordb.FalkorDB(**conn_kwargs).select_graph(_HUB_FACTS_GRAPH)

    if graph.query("MATCH (f:Fact {content_hash: $h}) RETURN f LIMIT 1", {"h": fact_id}).result_set:
        return {"success": True, "message": "Fact already exists (sha256 dedup skipped)", "skipped": True}

    graph.query(
        "CREATE (:Fact {content_hash: $h, content: $t, session_id: $s, spoke: $sp, "
        "created_at: $c, created_at_iso: $ci, superseded_by: ''})",
        {"h": fact_id, "t": content, "s": session_id, "sp": SPOKE_ID, "c": now, "ci": now_iso},
    )

    db = _lancedb.connect(_LANCEDB_PATH)
    existing_tables = db.list_tables()
    if _HUB_FACTS_TABLE in existing_tables.tables:
        tbl = db.open_table(_HUB_FACTS_TABLE)
    else:
        import pyarrow as _pa
        schema = _pa.schema([
            _pa.field("fact_id",    _pa.string()),
            _pa.field("fact_text",  _pa.string()),
            _pa.field("spoke",      _pa.string()),
            _pa.field("superseded", _pa.bool_()),
            _pa.field("vector",     _pa.list_(_pa.float32(), _HUB_FACTS_DIM)),
        ])
        tbl = db.create_table(_HUB_FACTS_TABLE, schema=schema)
    tbl.add([{"fact_id": fact_id, "fact_text": content, "spoke": SPOKE_ID,
              "superseded": False, "vector": vector}])
    return {"success": True, "message": "Fact written to hub_facts", "skipped": False}


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@mcp.tool(annotations=ToolAnnotations(title="Store Memory", destructiveHint=False))
async def memory_store(
    content: str,
    tags: Optional[str] = None,
    memory_type: Optional[str] = None,
    context: Optional[str] = None,
    supersedes_id: Optional[str] = None,
) -> MemoryStoreResult:
    """Store a memory in the namespace-scoped FalkorDB memory store.

    Args:
        content: The text to remember — fact, decision, finding, or context snippet.
        tags: Optional comma-separated tags (e.g. "decision,architecture").
              The spoke tag (spoke:<SPOKE_ID>) is always injected automatically.
        memory_type: Optional ontology type — observation, decision, planning, reference,
                     architecture, pattern, milestone, note, etc.
        context: Optional extra context stored in metadata.
        supersedes_id: Optional content_hash of an existing Memory node to invalidate.
    """
    content = content.strip()
    if not content:
        return MemoryStoreResult(success=False, message="Content cannot be empty", spoke=SPOKE_ID)

    storage = await _get_storage()
    tag_list = _inject_spoke_tag(_parse_tags(tags))
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    metadata: dict = {"spoke": SPOKE_ID}
    if context:
        metadata["context"] = context

    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=tag_list,
        memory_type=memory_type or "observation",
        metadata=metadata,
    )

    try:
        success, message = await storage.store(memory)
        if success and supersedes_id:
            try:
                import falkordb as _falkordb
                host = os.environ.get("FALKORDB_HOST", "localhost")
                port = int(os.environ.get("FALKORDB_PORT", "6379"))
                password = os.environ.get("FALKORDB_PASSWORD", "")
                conn_kwargs: dict = {"host": host, "port": port}
                if password:
                    conn_kwargs["password"] = password
                graph = _falkordb.FalkorDB(**conn_kwargs).select_graph(f"memory_{SPOKE_ID}")
                graph.query("MATCH (m:Memory {content_hash: $h}) SET m.invalidated = 1", {"h": supersedes_id})
            except Exception as inv_exc:
                logger.warning("memory_store: failed to invalidate prior node %s: %s", supersedes_id[:8], inv_exc)
        return MemoryStoreResult(success=success, message=message, spoke=SPOKE_ID)
    except Exception as e:
        logger.error("memory_store error: %s", e, exc_info=True)
        return MemoryStoreResult(success=False, message=str(e), spoke=SPOKE_ID)


@mcp.tool(annotations=ToolAnnotations(title="Search Memories", destructiveHint=False))
async def memory_search(
    query: str,
    limit: int = 10,
    tags: Optional[str] = None,
    quality_boost: float = 0.0,
) -> MemorySearchResult:
    """Search namespace-scoped memories by semantic similarity.

    Args:
        query: Natural language search query.
        limit: Max results to return (default 10).
        tags: Optional comma-separated tags to narrow results further.
        quality_boost: 0.0 = pure semantic (default); 0.3 = 30% quality weight.
    """
    query = query.strip()
    if not query:
        return MemorySearchResult(memories=[], spoke=SPOKE_ID, error="Query cannot be empty")
    if not 0.0 <= quality_boost <= 1.0:
        return MemorySearchResult(memories=[], spoke=SPOKE_ID, error="quality_boost must be between 0.0 and 1.0")

    storage = await _get_storage()
    limit = _clamp_limit(limit)
    filter_tags = _inject_spoke_tag(_parse_tags(tags))

    try:
        if quality_boost > 0.0:
            results = await storage.retrieve_with_quality_boost(
                query=query, n_results=limit, tags=filter_tags,
                quality_boost=True, quality_weight=quality_boost,
            )
        else:
            results = await storage.retrieve(query=query, n_results=limit, tags=filter_tags)

        memories = []
        for r in results:
            m = r.memory
            memories.append(MemoryItem(
                content_hash=m.content_hash,
                content=m.content,
                tags=m.tags,
                score=round(r.relevance_score, 4) if r.relevance_score else None,
                memory_type=m.memory_type,
                created_at=_normalize_iso_timestamp(m),
                spoke=SPOKE_ID,
            ))
        return MemorySearchResult(memories=memories, spoke=SPOKE_ID)
    except Exception as e:
        logger.error("memory_search error: %s", e, exc_info=True)
        return MemorySearchResult(memories=[], spoke=SPOKE_ID, error=str(e))


@mcp.tool(annotations=ToolAnnotations(title="Wakeup — Restore Session Context", destructiveHint=False))
async def memory_wakeup(focus_hint: Optional[str] = None) -> MemoryWakeupResult:
    """Restore session context from FalkorDB namespace memory.

    Args:
        focus_hint: Optional topic to bias fallback semantic queries.
    """
    storage = await _get_storage()
    focus_hint = (focus_hint or "").strip()
    lookback_days = _wakeup_lookback_days()
    time_cutoff = time.time() - (lookback_days * 24 * 60 * 60)

    try:
        milestones, decisions, focus_updates, blockers = await asyncio.gather(
            storage.search_by_tags([_SPOKE_TAG, "milestone"], operation="AND", time_start=time_cutoff),
            storage.search_by_tags([_SPOKE_TAG, "decision"], operation="AND", time_start=time_cutoff),
            storage.search_by_tags([_SPOKE_TAG, "focus"], operation="AND", time_start=time_cutoff),
            storage.search_by_tags([_SPOKE_TAG, "blocker"], operation="AND", time_start=time_cutoff),
        )

        if not any((milestones, decisions, focus_updates, blockers)):
            fallback_query = focus_hint or "recent tasks decisions blockers session context"
            fallback_results = await storage.retrieve(query=fallback_query, n_results=WAKEUP_FALLBACK_LIMIT, tags=[_SPOKE_TAG])
            filtered = [r for r in fallback_results if _is_recent_memory(r.memory, time_cutoff)]
            groups = {"active_tasks": [], "recent_decisions": [], "blockers": []}
            for result in filtered:
                groups[_categorize_fallback_memory(result.memory)].append(result.memory)
            active_combined = sorted(
                _dedupe_memories([m for m in groups["active_tasks"] if not _is_session_close_memory(m)]),
                key=_memory_sort_key, reverse=True,
            )
            return MemoryWakeupResult(
                spoke=SPOKE_ID,
                active_tasks=_format_wakeup_memories(active_combined),
                recent_decisions=_format_wakeup_memories(groups["recent_decisions"]),
                blockers=_format_wakeup_memories(groups["blockers"]) or [WakeupMemoryItem(content="None", score=None)],
            )

        active_combined = sorted(
            _dedupe_memories([m for m in (milestones + focus_updates) if not _is_session_close_memory(m)]),
            key=_memory_sort_key, reverse=True,
        )
        return MemoryWakeupResult(
            spoke=SPOKE_ID,
            active_tasks=_format_wakeup_memories(active_combined),
            recent_decisions=_format_wakeup_memories(decisions),
            blockers=_format_wakeup_memories(blockers) if blockers else [WakeupMemoryItem(content="None", score=None)],
        )
    except Exception as e:
        logger.error("memory_wakeup error: %s", e, exc_info=True)
        return MemoryWakeupResult(spoke=SPOKE_ID, active_tasks=[], recent_decisions=[],
                                  blockers=[WakeupMemoryItem(content="None", score=None)], error=str(e))


@mcp.tool(annotations=ToolAnnotations(title="Close Session", destructiveHint=False))
async def memory_close_session(session_id: Optional[str] = None, summary: Optional[str] = None) -> SessionCloseResult:
    """Close current session and store session metadata in FalkorDB.

    Args:
        session_id: Optional session identifier.
        summary: Optional one-line session summary.
    """
    storage = await _get_storage()
    now = time.time()
    now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    session_metadata = {"session_id": session_id or "unknown", "closed_at": now_iso, "spoke": SPOKE_ID}
    if summary:
        session_metadata["summary"] = summary

    content = f"Session closed: {session_id or 'unknown'}" + (f" — {summary}" if summary else "")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    tags = _inject_spoke_tag(["session-close", "milestone", session_id] if session_id else ["session-close", "milestone"])

    memory = Memory(content=content, content_hash=content_hash, tags=tags,
                    memory_type="milestone", metadata=session_metadata)
    try:
        success, message = await storage.store(memory)
        return SessionCloseResult(success=success, message=message, spoke=SPOKE_ID, session_metadata=session_metadata)
    except Exception as e:
        logger.error("memory_close_session error: %s", e, exc_info=True)
        return SessionCloseResult(success=False, message=str(e), spoke=SPOKE_ID, session_metadata=session_metadata)


@mcp.tool(annotations=ToolAnnotations(title="Delete Memory", destructiveHint=True))
async def memory_delete(content_hash: str, confirm: bool, reason: str, dry_run: bool = False) -> MemoryDeleteResult:
    """Delete a memory by content hash. Requires ENABLE_MEMORY_DELETE=true.

    Args:
        content_hash: SHA-256 hash of the memory content to delete.
        confirm: Must be true for non-dry-run deletion.
        reason: Human-readable reason for audit logging.
        dry_run: If true, validates without deleting.
    """
    normalized_hash = content_hash.strip().lower()
    reason = reason.strip()

    if not _memory_delete_enabled():
        return MemoryDeleteResult(success=False, message="memory_delete is disabled. Set ENABLE_MEMORY_DELETE=true to enable.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=dry_run, error="disabled")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized_hash):
        return MemoryDeleteResult(success=False, message="content_hash must be a 64-character lowercase hex SHA-256 value.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=dry_run, error="invalid_hash")
    if not reason:
        return MemoryDeleteResult(success=False, message="reason is required.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=dry_run, error="missing_reason")
    if not dry_run and not confirm:
        return MemoryDeleteResult(success=False, message="confirm=true is required for deletion.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=False, error="confirmation_required")

    storage = await _get_storage()
    try:
        existing = await storage.get_by_hash(normalized_hash)
    except Exception as e:
        return MemoryDeleteResult(success=False, message="Failed to validate memory existence.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=dry_run, error=str(e))

    if not existing:
        return MemoryDeleteResult(success=False, message="Memory not found.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=dry_run, error="not_found")
    if _SPOKE_TAG not in set(existing.tags or []):
        return MemoryDeleteResult(success=False, message="Refusing deletion: memory does not belong to current namespace.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=dry_run, error="spoke_mismatch")
    if dry_run:
        return MemoryDeleteResult(success=True, message="Dry run successful. Memory is eligible for deletion.",
                                  spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=True)

    success, message = await storage.delete(normalized_hash)
    if not success:
        return MemoryDeleteResult(success=False, message=message, spoke=SPOKE_ID,
                                  content_hash=normalized_hash, dry_run=False, error="delete_failed")

    try:
        audit_ts = datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat()
        audit_content = f"memory_delete: deleted {normalized_hash} in {SPOKE_ID} at {audit_ts}; reason={reason}"
        audit_hash = hashlib.sha256(f"{audit_content}|{audit_ts}".encode("utf-8")).hexdigest()
        audit_memory = Memory(content=audit_content, content_hash=audit_hash,
                              tags=_inject_spoke_tag(["audit", "memory-delete", "milestone"]),
                              memory_type="milestone",
                              metadata={"spoke": SPOKE_ID, "deleted_content_hash": normalized_hash,
                                        "reason": reason, "action": "memory_delete", "timestamp": audit_ts})
        await storage.store(audit_memory)
    except Exception as e:
        logger.warning("memory_delete audit write failed: %s", e)

    return MemoryDeleteResult(success=True, message="Memory deleted successfully.",
                              spoke=SPOKE_ID, content_hash=normalized_hash, dry_run=False, deleted=True)


@mcp.tool(annotations=ToolAnnotations(title="Recall Facts", destructiveHint=False))
async def memory_recall(query: str, limit: int = 10) -> FactRecallResult:
    """Recall current atomic facts about a topic from the persistent facts store.

    Args:
        query: Natural language question.
        limit: Max results to return (default 10, max 20).
    """
    query = query.strip()
    if not query:
        return FactRecallResult(facts=[], spoke=SPOKE_ID, query=query, error="Query cannot be empty")
    clamped = max(1, min(limit, _HUB_FACTS_LIMIT))
    try:
        recaller = _HubFactsRecaller.get()
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, lambda: recaller.query(query, include_superseded=False, limit=clamped))
        return FactRecallResult(facts=facts, spoke=SPOKE_ID, query=query, include_superseded=False)
    except Exception as e:
        logger.error("memory_recall error: %s", e, exc_info=True)
        return FactRecallResult(facts=[], spoke=SPOKE_ID, query=query, include_superseded=False, error=str(e))


@mcp.tool(annotations=ToolAnnotations(title="Recall Facts History", destructiveHint=False))
async def memory_recall_history(query: str, limit: int = 10) -> FactRecallResult:
    """Recall full history of facts including superseded versions.

    Args:
        query: Natural language question.
        limit: Max results to return (default 10, max 20).
    """
    query = query.strip()
    if not query:
        return FactRecallResult(facts=[], spoke=SPOKE_ID, query=query, include_superseded=True, error="Query cannot be empty")
    clamped = max(1, min(limit, _HUB_FACTS_LIMIT))
    try:
        recaller = _HubFactsRecaller.get()
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, lambda: recaller.query(query, include_superseded=True, limit=clamped))
        return FactRecallResult(facts=facts, spoke=SPOKE_ID, query=query, include_superseded=True)
    except Exception as e:
        logger.error("memory_recall_history error: %s", e, exc_info=True)
        return FactRecallResult(facts=[], spoke=SPOKE_ID, query=query, include_superseded=True, error=str(e))


@mcp.tool(annotations=ToolAnnotations(title="Store Fact", destructiveHint=False))
async def memory_store_fact(content: str, session_id: Optional[str] = None) -> FactStoreResult:
    """Write a single atomic fact to the persistent facts store (FalkorDB + LanceDB).

    Args:
        content: Self-contained, present-tense fact statement.
        session_id: Optional provenance label (default: "manual").
    """
    content = content.strip()
    if not content:
        return FactStoreResult(success=False, fact_id="", message="Content cannot be empty", spoke=SPOKE_ID)

    fact_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
    sid = (session_id or "manual").strip()
    now = time.time()
    now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _write_hub_fact(content, fact_id, sid, now, now_iso))
        return FactStoreResult(success=result["success"], fact_id=fact_id[:12],
                               message=result["message"], spoke=SPOKE_ID, skipped=result.get("skipped", False))
    except Exception as e:
        logger.error("memory_store_fact error: %s", e, exc_info=True)
        return FactStoreResult(success=False, fact_id=fact_id[:12], message=str(e), spoke=SPOKE_ID)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    """Entry point for gernash-memory-server console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
