from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base_event import BaseEvent


class SearchJobCreatedEvent(BaseEvent):
    """
    Entry point for search pipeline.
    Emitted when a new search job is saved via the Search API.
    """

    event_type: str = "search_job.created"
    query_job_id: str
    query: str  # Carried inline so SearchDispatcher can create EmbedTask without a DB lookup
    embedding_profile: str  # Must match the model used to embed the chunks
    rerank: bool = False
    rerank_profile: Optional[str] = None  # Key into RERANK_CONFIGS; None uses "default"


class SearchQueryEmbeddedEvent(BaseEvent):
    """
    Emitted by GpuEmbedWorker after a search query vector is written.
    Drives: cpu.search.run task via SearchDispatcher (second hop).
    """

    event_type: str = "search_query.embedded"
    query_job_id: str
    model_name: str


class CandidateChunkRef(BaseModel):
    """One ANN candidate carried through the rerank hop: id + its ANN score.

    Embedded in SearchCandidatesRetrievedEvent and SearchRankTask /
    SearchFinalizeTask so the original vector-similarity score survives even
    when a job skips reranking (SearchFinalizeHandler still needs *some*
    score to report, and RerankHandler reports the reranked score instead).
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float


class SearchCandidatesRetrievedEvent(BaseEvent):
    """
    Emitted by SearchHandler after the ANN candidate pool is fetched —
    unconditionally, whether or not the job asked for reranking. Keeps the
    event log uniform across both paths: replay/audit doesn't need to know
    which branch a given job took.

    Drives: SearchRankTask (rerank=True) or SearchFinalizeTask (rerank=False)
    via SearchDispatcher's second search subscription.
    """

    event_type: str = "search_candidates.retrieved"
    query_job_id: str
    candidates: List[CandidateChunkRef]
    k: int
    rerank: bool
    rerank_profile: Optional[str] = None


class SearchJobCompletedEvent(BaseEvent):
    """
    Final search results ready.
    """

    event_type: str = "search_job.completed"
    query_job_id: str