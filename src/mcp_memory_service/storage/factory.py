# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared storage backend factory for the MCP Memory Service.

This module provides a single, shared factory function for creating storage backends,
eliminating code duplication between the MCP server and web interface initialization.
"""

import logging
from typing import Type

from .base import MemoryStorage

logger = logging.getLogger(__name__)


def get_storage_backend_class() -> Type[MemoryStorage]:
    """
    Get storage backend class based on configuration.

    Returns:
        Storage backend class
    """
    from ..config import STORAGE_BACKEND

    backend = STORAGE_BACKEND.lower()

    if backend == "falkordb":
        from .falkordb import FalkorDBMemoryStorage
        return FalkorDBMemoryStorage
    elif backend == "cloudflare":
        try:
            from .cloudflare import CloudflareStorage
            return CloudflareStorage
        except ImportError as e:
            logger.error(f"Failed to import Cloudflare storage: {e}")
            raise
    else:
        raise ValueError(f"Unsupported storage backend: '{backend}'")


async def create_storage_instance(server_type: str = None) -> MemoryStorage:
    """
    Create and initialize storage backend instance based on configuration.

    Args:
        server_type: Optional server type identifier ("mcp" or "http")

    Returns:
        Initialized storage backend instance
    """
    from ..config import (
        STORAGE_BACKEND, EMBEDDING_MODEL_NAME,
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
        CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
        CLOUDFLARE_BASE_DELAY,
    )

    logger.info(f"Creating storage backend instance (backend: {STORAGE_BACKEND}, server_type: {server_type})...")

    StorageClass = get_storage_backend_class()

    if StorageClass.__name__ == "FalkorDBMemoryStorage":
        storage = StorageClass(embedding_model=EMBEDDING_MODEL_NAME)
        logger.info("Initialized FalkorDB storage")

    elif StorageClass.__name__ == "CloudflareStorage":
        storage = StorageClass(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
            r2_bucket=CLOUDFLARE_R2_BUCKET,
            embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
            large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            max_retries=CLOUDFLARE_MAX_RETRIES,
            base_delay=CLOUDFLARE_BASE_DELAY
        )
        logger.info(f"Initialized Cloudflare storage with vectorize index: {CLOUDFLARE_VECTORIZE_INDEX}")

    else:
        raise ValueError(f"Unsupported storage backend class: {StorageClass.__name__}")

    await storage.initialize()
    logger.info(f"Storage backend {StorageClass.__name__} initialized successfully")

    return storage
