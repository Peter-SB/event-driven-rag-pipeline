def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in table names.

    Replaces hyphens and dots with underscores.
    E.g., "bge-base-v1.5" → "bge_base_v1_5"
    """
    model = model.lower()
    return model.replace("/", "_").replace("-", "_").replace(".", "_")

def build_chunk_table_name(post_table: str, task_type: str, model: str) -> str:
    """Build a consistent table name for storing chunks.

    Format: {post_table}_chunks_{task_type}_{model_sanitized}
    Example: posts_main_chunks_body_bge_base_v1_5

    Args:
        post_table: The library's post table (e.g., "posts_main")
        task_type: The chunk type (e.g., "body", "summary_title", "analysis")
        model: The embedding model name (e.g., "bge-base-v1.5")
    """
    # Sanitize model name: hyphens and dots → underscores
    # e.g. "bge-base-v1.5" → "bge_base_v1_5", "qwen3-0.6b" → "qwen3_0_6b"
    model_safe = sanitize_model_name(model)

    return f"{post_table}_chunks_{task_type.lower()}_{model_safe}"



# ---------------------------------------------------------------------------
# Chunk table reverse-parsing
# ---------------------------------------------------------------------------

def build_chunk_table_suffix_map(
    embed_configs: dict | None = None,
) -> dict[str, tuple[str, object]]:
    """Build a reverse lookup from chunk table suffix → (task_type, EmbedConfig).

    Chunk table names follow the pattern::

        posts_{library}_chunks_{task_type}_{model_sanitized}

    The part after ``_chunks_`` is ``{task_type}_{model_sanitized}``.  This
    function pre-computes that suffix for every entry in EMBED_CONFIGS so the
    caller can reverse-parse any chunk table name found in the database.

    Args:
        embed_configs: Override the default EMBED_CONFIGS (useful for testing).

    Returns:
        A dict like ``{"body_baai_bge_base_en_v1_5": ("body", EmbedConfig(...))}``,
        keyed by the post-``_chunks_`` part of the table name.
    """
    from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS as _DEFAULT

    configs = embed_configs if embed_configs is not None else _DEFAULT
    return {
        f"{task_type}_{sanitize_model_name(cfg.model)}": (task_type, cfg)
        for task_type, cfg in configs.items()
    }


def parse_chunk_table_name(
    table_name: str,
    suffix_map: dict[str, tuple[str, object]],
) -> tuple[str, str, object] | None:
    """Parse a chunk table name into ``(post_table, task_type, embed_cfg)``.

    Returns ``None`` if the table name doesn't match any entry in *suffix_map*
    (e.g. hand-created tables or stale tables from a removed config entry).

    Args:
        table_name: A raw table name, e.g. ``posts_main_chunks_body_baai_bge_base_en_v1_5``.
        suffix_map: A map built by :func:`build_chunk_table_suffix_map`.

    Returns:
        ``(post_table, task_type, embed_cfg)`` or ``None``.
    """
    marker = "_chunks_"
    idx = table_name.find(marker)
    if idx == -1:
        return None

    post_table = table_name[:idx]
    suffix = table_name[idx + len(marker):]

    match = suffix_map.get(suffix)
    if match is None:
        return None

    task_type, embed_cfg = match
    return post_table, task_type, embed_cfg


def build_search_job_table_name(post_table: str, task_type: str, model: str) -> str:
    """Build a consistent table name for storing search jobs.

    Format: {post_table}_search_jobs_{task_type}_{model_sanitized}
    Example: posts_main_search_jobs_summary_title_bge_base_v1_5
    
    Args:
        post_table: The library's post table (e.g., "posts_main")
        task_type: The search job type (e.g., "summary_title")
        model: The embedding model name (e.g., "bge-base-v1.5")
    """
    model_safe = sanitize_model_name(model)
    return f"{post_table}_search_jobs_{task_type.lower()}_{model_safe}"
