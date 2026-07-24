"""Shared Pydantic field validators reused across the API request models.

Both request validation and error messages must stay identical between
search, similar, and sync — duplicating the regex/logic risks the
messages drifting out of sync (as happened previously with sync.py's
library_id message).
"""
from __future__ import annotations

import re

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS

LIBRARY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_library_id(v: str) -> str:
    if not v or not LIBRARY_ID_PATTERN.match(v):
        raise ValueError("library_id must start with '[a-z]' and contain only '[a-z0-9_]'")
    return v


def validate_chunk_type(v: str) -> str:
    if v not in EMBED_CONFIGS:
        valid = list(EMBED_CONFIGS.keys())
        raise ValueError(f"chunk_type must be one of {valid}, got {v!r}")
    return v
