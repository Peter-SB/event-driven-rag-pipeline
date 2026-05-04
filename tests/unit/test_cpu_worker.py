"""Tests for the _build_chunks helper in chunk_handler.

These are pure unit tests with no I/O — they verify that the chunking utility
produces correctly structured Chunk objects.

For handler business logic tests (text routing, idempotency, event publishing)
see test_chunk_handler.py.

Tested behaviours
-----------------
- Chunks have the correct text_hash (sha256 of window text)
- token_count is word-based (words * 1.3, minimum 1)
- chunk_index starts at 0 and increments sequentially
- All chunks carry the same post_id
"""
from __future__ import annotations

import hashlib

import pytest

from event_driven_rag_service.handlers.chunk_handler import _build_chunks


# ---------------------------------------------------------------------------
# _build_chunks helper unit tests
# ---------------------------------------------------------------------------

def test_build_chunks_produces_chunks_with_text_hash():
    chunks = _build_chunks(1, "2024-01-01T00:00:00+00:00", "Hello world " * 50, "My Title")
    assert len(chunks) >= 1
    for c in chunks:
        expected = hashlib.sha256(c.text.encode()).hexdigest()
        assert c.text_hash == expected


def test_build_chunks_token_count_is_word_based():
    text = "word " * 100  # 100 words
    chunks = _build_chunks(1, "2024-01-01T00:00:00+00:00", text, None)
    for c in chunks:
        words_in_chunk = len(c.text.split())
        expected_tokens = max(1, round(words_in_chunk * 1.3))
        assert c.token_count == expected_tokens


def test_build_chunks_index_starts_at_zero_and_increments():
    text = "paragraph\n\n".join(["word " * 200 for _ in range(4)])
    chunks = _build_chunks(1, "2024-01-01T00:00:00+00:00", text, None)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_build_chunks_all_have_same_post_id():
    text = "word " * 300
    chunks = _build_chunks(42, "2024-01-01T00:00:00+00:00", text, None)
    assert all(c.post_id == 42 for c in chunks)

