def build_chunk_table_name(field: str, model: str) -> str:
    """Build a consistent table name for storing chunks from a given source."""
    
    # todo: proper sanitization/validation to prevent SQL injection, reserved words, etc.
    model = model.replace("-", "_").replace(".", "_")  # e.g. "bge-base-v1.5" → "bge_base_v1_5"

    return f"chunks_{field.lower()}_{model.lower()}"