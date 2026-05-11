class ChunkTableNotFoundError(Exception):
    """Raised when the chunk table referenced by an EmbedTask does not exist.

    This is unrecoverable via retry — the CPU worker never created the table
    (likely failed before doing so). The task should be skipped, not DLQ'd.
    """
