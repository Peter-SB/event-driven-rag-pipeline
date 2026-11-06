# Machine-readable code for "the remote reranker could not be used". Sent as the
# `detail.code` of POST /search's 503 and stored on a failed search job
# (`error_code`), so a client handles both the up-front and the mid-job case the
# same way: offer a plain semantic search instead.
RERANK_UNAVAILABLE = "rerank_unavailable"


class ChunkTableNotFoundError(Exception):
    """Raised when the chunk table referenced by an EmbedTask does not exist.

    This is unrecoverable via retry — the CPU worker never created the table
    (likely failed before doing so). The task should be skipped, not DLQ'd.
    """


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no permitted embedding leg could serve a batch.

    Specifically: the remote endpoint is down (or does not serve this model)
    and the model is marked ``allow_local=False`` in EMBED_CONFIGS, so there is
    nowhere left to run it.

    Retrying is pointless until the endpoint is back, and falling through to a
    different model would poison the chunk table's pgvector column with vectors
    from another embedding space. So the batch is *skipped*: the chunk rows keep
    ``embedding IS NULL`` and ``POST /maintenance/requeue-missing-embeddings``
    re-queues them later. Subclasses RuntimeError so callers that only care
    "encode failed" keep working.
    """
