from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk."""

    title: Optional[str] = None
    external_id: Optional[str] = None  # source-specific ID (e.g. Reddit post ID)


class Chunk(BaseModel):
    """
    Represents a single text window produced by the chunker.

    Chunks are persisted by CpuChunkWorker before embedding. The embedding
    column is populated later by GpuEmbedWorker via an UPDATE — it is not
    part of this model.
    """

    id: str
    post_id: int
    post_updated_at: datetime
    chunk_index: int
    text: str
    metadata: ChunkMetadata
    token_count: int          # estimated from word count (word_count * 1.3)
    text_hash: str
    created_at: datetime
