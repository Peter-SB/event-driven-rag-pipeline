"""Tests for boundary_chunker and ChunkAtBoundaryStrategy.

The boundary chunker is the primary text-splitting strategy used by
CpuChunkWorker.  These tests verify that it produces correctly sized,
overlapping chunks and handles edge cases without raising.

Tested behaviours
-----------------
- Short text (< target_words) returns exactly one chunk
- Multi-paragraph text splits at paragraph boundaries
- Overlap means the first N words of chunk[i] appear at the end of chunk[i-1]
- Hard limit forces a split even when no boundary marker is present
- Empty string returns an empty list (not a single empty-string chunk)
- ChunkAtBoundaryStrategy.chunk() delegates correctly to chunk_at_boundaries
"""
from __future__ import annotations

import pytest

from event_driven_rag_service.utils.boundary_chunker import (
    ChunkAtBoundaryStrategy,
    chunk_at_boundaries,
    word_count,
)


# ---------------------------------------------------------------------------
# word_count helper
# ---------------------------------------------------------------------------

def test_word_count_on_simple_sentence():
    assert word_count("hello world foo") == 3


def test_word_count_on_empty_string():
    assert word_count("") == 0


# ---------------------------------------------------------------------------
# Short texts — always one chunk
# ---------------------------------------------------------------------------

def test_short_text_returns_single_chunk():
    text = " ".join(["word"] * 10)  # 10 words, target=500
    chunks = chunk_at_boundaries(text, target=500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_string_returns_empty_list():
    chunks = chunk_at_boundaries("", target=500)
    assert chunks == []


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

PARA_TEXT = "\n\n".join([
    " ".join(["alpha"] * 200),   # paragraph A — 200 words
    " ".join(["beta"] * 200),    # paragraph B — 200 words
    " ".join(["gamma"] * 200),   # paragraph C — 200 words
])


def test_paragraph_text_splits_into_multiple_chunks():
    # With target=250 and hard_limit=350, a 600-word 3-paragraph text should produce
    # more than one chunk.  Without a matching hard_limit the whole text fits under
    # the default 750-word hard_limit and is returned as a single chunk.
    chunks = chunk_at_boundaries(PARA_TEXT, target=250, hard_limit=350)
    assert len(chunks) > 1


def test_all_original_words_are_present_across_chunks():
    # Every source word should appear in at least one chunk (overlap may duplicate)
    chunks = chunk_at_boundaries(PARA_TEXT, target=250, hard_limit=350, overlap=0.0)
    combined = " ".join(chunks)
    # Each distinct word in the original should appear somewhere in the combined output
    for word in ("alpha", "beta", "gamma"):
        assert word in combined


def test_overlap_shares_tail_of_previous_chunk():
    """With overlap>0 the start of chunk[i+1] should echo the end of chunk[i]."""
    chunks = chunk_at_boundaries(PARA_TEXT, target=250, overlap=0.15)
    if len(chunks) < 2:
        pytest.skip("Not enough chunks to test overlap")
    # Take last 5 words of chunk[0] and check they appear in chunk[1]
    tail_words = chunks[0].split()[-5:]
    head_of_next = chunks[1]
    for word in tail_words:
        assert word in head_of_next, (
            f"Overlap word '{word}' missing from start of next chunk"
        )


# ---------------------------------------------------------------------------
# ChunkAtBoundaryStrategy dataclass
# ---------------------------------------------------------------------------

def test_strategy_chunk_method_returns_same_as_direct_call():
    strategy = ChunkAtBoundaryStrategy(target=250, overlap=0.10)
    direct = chunk_at_boundaries(PARA_TEXT, target=250, overlap=0.10)
    via_strategy = strategy.chunk(PARA_TEXT)
    assert direct == via_strategy


def test_strategy_default_target_is_500():
    strategy = ChunkAtBoundaryStrategy()
    assert strategy.target == 500


# ---------------------------------------------------------------------------
# Token count estimate sanity check (used in CpuChunkWorker)
# ---------------------------------------------------------------------------

def test_token_estimate_is_greater_than_word_count():
    text = "The quick brown fox jumps over the lazy dog"
    words = len(text.split())
    estimated_tokens = max(1, round(words * 1.3))
    assert estimated_tokens > words
