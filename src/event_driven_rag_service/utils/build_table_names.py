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
