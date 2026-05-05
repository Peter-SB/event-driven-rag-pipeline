"""
Chunk task: slice a post's text into overlapping token windows and persist the
chunks so the embedding pipeline can pick them up.

task_type variants
------------------
body           — chunk the post body (custom_body if present, else body_text)
summary_title  — chunk the title + summary as a single unit (preferred for
                 semantic search; keeps title context with the summary text)
analysis       — chunk an inference/analysis result; text is carried inline via
                 ``analysis_text`` (event-carried state transfer)
"""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from typing import Literal, Optional

from .base_task import BaseTask
from event_driven_rag_service.utils.chunk_strategies import chunk_at_boundaries


class ChunkTask(BaseTask):
    kind: Literal["chunk"] = "chunk"
    task_type: Literal["body", "summary_title", "analysis"]
    post_id: int
    post_table: str

    # Which model will embed these chunks — determines the chunk table name
    # so the GPU worker knows where to write embeddings.
    embed_model: str

    # Inline text for analysis tasks (event-carried state transfer).
    # Only populated when task_type="analysis"; worker falls back to post column
    # if absent.
    analysis_text: Optional[str] = None  # todo: remove and dont carry text in the event, check chunk database

    def chunk_table_name(self) -> str:
        """Derive the target Postgres table name for this task's chunks."""
        from event_driven_rag_service.utils.build_table_names import build_chunk_table_name
        return build_chunk_table_name(self.post_table, self.task_type, self.embed_model)


@dataclass
class ChunkAtBoundaryStrategy:
    """Configuration for boundary-aware chunking.

    Pass an instance as ``ChunkTask.strategy`` to control how text is split.
    Call :meth:`chunk` to produce the list of chunk strings.
    """

    target: int = 500
    chunk_size_tolerance: float = 0.20
    hard_limit: int = 750
    overlap: float = 0.30
    start_offset: float = 0.0

    def chunk(self, text: str) -> list[str]:
        """Run the boundary chunker and return the list of chunk strings."""
        return chunk_at_boundaries(
            text,
            target=self.target,
            chunk_size_tolerance=self.chunk_size_tolerance,
            hard_limit=self.hard_limit,
            overlap=self.overlap,
            start_offset=self.start_offset,
        )

    def to_dict(self) -> dict:
        """Serialise strategy parameters to a plain dict."""
        return dataclasses.asdict(self)
