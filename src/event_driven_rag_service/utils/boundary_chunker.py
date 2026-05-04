"""boundary_chunker.py — Boundary-aware text chunker.

Splits text into chunks by seeking natural break markers rather than
cutting naively at a character offset.

Break marker hierarchy (preferred → fallback):
  "***"  >  "---"  >  "\\n\\n"  >  "\\n"  >  "."

Parameters
----------
target                : desired chunk size in words
chunk_size_tolerance  : fractional ± window around target (e.g. 0.20 → ±20%)
hard_limit            : absolute maximum words per chunk before a forced split
overlap               : fraction of the previous chunk to re-include at the
                        start of the next chunk (e.g. 0.10 → trailing ~10%)
start_offset          : fraction of the whole text to skip before chunking
                        begins (e.g. 0.25 → start a quarter of the way in)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

MARKERS = ["***", "---", "\n\n", "\n", "."]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def find_breaks(text: str, marker: str) -> list[int]:
    """Return all character positions immediately after each occurrence of `marker`."""
    positions = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        positions.append(idx + len(marker))
        start = idx + 1
    return positions


def nearest_break(
    text: str,
    target_char: int,
    search_window: tuple[int, int],
    markers: list[str],
) -> int | None:
    """Find the break position closest to `target_char` within the window.

    Iterates the marker hierarchy; the first marker that has any candidate
    inside [lo, hi] wins.  Returns None when no marker is found.
    """
    lo, hi = search_window
    for marker in markers:
        candidates = [p for p in find_breaks(text, marker) if lo <= p <= hi]
        if candidates:
            return min(candidates, key=lambda p: abs(p - target_char))
    return None


def char_for_word_fraction(text: str, fraction: float) -> int:
    """Return the character index corresponding to `fraction` of total word count.

    Clamps to [0, len(text)].
    """
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return len(text)

    words = text.split()
    target_word = int(len(words) * fraction)
    count = 0
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        while i < len(text) and not text[i].isspace():
            i += 1
        count += 1
        if count >= target_word:
            break
    return i


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_at_boundaries(
    text: str,
    target: int = 500,
    chunk_size_tolerance: float = 0.20,
    hard_limit: int = 750,
    overlap: float = 0.3,
    start_offset: float = 0.0,
) -> list[str]:
    """Split `text` into boundary-aligned chunks.

    Parameters
    ----------
    text                  : source string to chunk
    target                : desired chunk size in words
    chunk_size_tolerance  : fractional ± window around target (e.g. 0.20 → ±20%)
    hard_limit            : absolute maximum words per chunk before a forced split
    overlap               : fraction of the previous chunk's words to prepend to
                            the next chunk (e.g. 0.10 → ~10% of last chunk recycled)
    start_offset          : fraction of the full text to skip before chunking
                            (e.g. 0.25 → begin ~25% of the way into the text)

    Returns
    -------
    List of stripped chunk strings.
    """
    # Normalise literal escape sequences that appear in raw text.
    text = text.replace(r"\n\n", "\n\n").replace(r"\n", "\n").replace(r"\r", "\r")

    # Apply start_offset: find the character position for the given word-fraction
    # then snap forward to the nearest clean break so we don't start mid-sentence.
    if start_offset > 0.0:
        offset_char = char_for_word_fraction(text, start_offset)
        snap_window = (offset_char, min(offset_char + 300, len(text)))
        snapped = nearest_break(text, offset_char, snap_window, ["\n\n", "\n", "."])
        cursor = snapped if snapped is not None else offset_char
    else:
        cursor = 0

    chunks: list[str] = []
    prev_chunk_text: str = ""

    while cursor < len(text):
        remaining = text[cursor:]

        # Build overlap prefix from the tail of the previous chunk, snapping to
        # the nearest marker when possible; fall back to a raw word-count slice.
        if overlap > 0.0 and prev_chunk_text:
            target_start = char_for_word_fraction(prev_chunk_text, 1.0 - overlap)
            slack = max(30, int(len(prev_chunk_text) * 0.10))
            lo = max(0, target_start - slack)
            hi = min(len(prev_chunk_text), target_start + slack)
            snap = nearest_break(prev_chunk_text, target_start, (lo, hi), MARKERS)
            if snap is not None:
                overlap_prefix = prev_chunk_text[snap:].strip() + " "
            else:
                prev_words = prev_chunk_text.split()
                overlap_word_count = max(1, int(len(prev_words) * overlap))
                overlap_prefix = " ".join(prev_words[-overlap_word_count:]) + " "
        else:
            overlap_prefix = ""

        # If everything remaining (plus overlap prefix) fits within hard_limit,
        # take it all as the final chunk.
        if word_count(overlap_prefix + remaining) <= hard_limit:
            chunk = (overlap_prefix + remaining).strip()
            if chunk:
                chunks.append(chunk)
            break

        words = remaining.split()
        avg_chars = len(remaining) / max(len(words), 1)

        overlap_words_added = word_count(overlap_prefix)
        effective_target = max(10, target - overlap_words_added)

        target_chars = int(effective_target * avg_chars)
        lo_chars = int(effective_target * (1 - chunk_size_tolerance) * avg_chars)
        hi_chars = int(
            min(effective_target * (1 + chunk_size_tolerance), hard_limit - overlap_words_added)
            * avg_chars
        )

        abs_target = cursor + target_chars
        abs_lo = cursor + lo_chars
        abs_hi = cursor + min(hi_chars, len(remaining))

        break_pos = nearest_break(
            text,
            target_char=abs_target,
            search_window=(abs_lo, abs_hi),
            markers=MARKERS,
        )

        if break_pos is not None:
            raw_chunk = text[cursor:break_pos].strip()
        else:
            # No clean break found — hard-split at a word boundary at hard_limit.
            hard_chars = int((hard_limit - overlap_words_added) * avg_chars)
            end = cursor + min(hard_chars, len(remaining))
            while end > cursor and text[end - 1] not in (" ", "\n"):
                end -= 1
            if end == cursor:
                end = cursor + hard_chars
            raw_chunk = text[cursor:end].strip()
            break_pos = end

        chunk = (overlap_prefix + raw_chunk).strip()
        if chunk:
            chunks.append(chunk)

        prev_chunk_text = raw_chunk
        cursor = break_pos

    return chunks


# ---------------------------------------------------------------------------
# Strategy class  (thin wrapper so callers can hold a configured instance)
# ---------------------------------------------------------------------------

@dataclass
class ChunkAtBoundaryStrategy:
    """Callable wrapper around ``chunk_at_boundaries`` with baked-in defaults.

    Use this when you need to pass a configured chunker as an object rather
    than calling ``chunk_at_boundaries`` directly.

    Example::

        strategy = ChunkAtBoundaryStrategy(target=400, overlap=0.15)
        windows = strategy.chunk(text)
    """

    target: int = 500
    chunk_size_tolerance: float = 0.20
    hard_limit: int = 750
    overlap: float = 0.10
    start_offset: float = 0.0

    def chunk(self, text: str) -> list[str]:
        return chunk_at_boundaries(
            text,
            target=self.target,
            chunk_size_tolerance=self.chunk_size_tolerance,
            hard_limit=self.hard_limit,
            overlap=self.overlap,
            start_offset=self.start_offset,
        )
