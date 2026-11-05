"""Unit tests for the playlist-recommendation service, using an in-memory reader."""
from __future__ import annotations

import numpy as np
import pytest

from event_driven_rag_service.repository.recommendation_repository import CandidateRow
from event_driven_rag_service.services.playlist_recommendation import (
    EmptyPlaylistError,
    RecommendParams,
    recommend_for_playlist,
)
from event_driven_rag_service.services.playlist_scoring import normalise

D = 32


class FakeReader:
    """Holds a small corpus; ``nearest_candidates`` is brute-force cosine."""

    def __init__(self, corpus: dict[int, np.ndarray], folders: dict[int, list[int]] | None = None):
        self.corpus = {i: normalise(v)[0] for i, v in corpus.items()}
        self.folders = folders or {}
        self.excludes: list[list[int]] = []

    async def folder_post_ids(self, posts_table, folder_id):
        return self.folders.get(folder_id, [])

    async def playlist_embeddings(self, chunks_table, post_ids):
        return {i: self.corpus[i] for i in post_ids if i in self.corpus}

    async def nearest_candidates(self, chunks_table, posts_table, query, limit, exclude_post_ids):
        self.excludes.append(list(exclude_post_ids))
        rows = [i for i in self.corpus if i not in set(exclude_post_ids)]
        rows.sort(key=lambda i: -float(self.corpus[i] @ query))
        return [CandidateRow(i, f"story {i}", "sub", None, self.corpus[i]) for i in rows[:limit]]

    async def post_titles(self, posts_table, post_ids):
        return {i: f"story {i}" for i in post_ids}


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    centres = normalise(rng.standard_normal((2, D)))
    out = {}
    for i in range(40):  # 0-19 theme A, 20-39 theme B
        out[i] = centres[i // 20] + 0.2 * rng.standard_normal(D)
    return out


async def run(reader, **kw):
    kw.setdefault("params", RecommendParams(k=5, diversity=0.0))
    return await recommend_for_playlist(reader, posts_table="posts_x", chunks_table="posts_x_chunks_y", **kw)


async def test_recommends_same_theme_and_never_the_playlist(corpus):
    reader = FakeReader(corpus)
    res = await run(reader, post_ids=list(range(0, 8)))
    got = [r.post_id for r in res.items]
    assert len(got) == 5 and not set(got) & set(range(0, 8))
    assert all(g < 20 for g in got)  # theme A neighbours, not theme B
    assert [r.rank for r in res.items] == [1, 2, 3, 4, 5]
    assert res.items[0].because and res.items[0].because[0].post_id < 8


async def test_folder_source_and_source_validation(corpus):
    reader = FakeReader(corpus, folders={1: list(range(20, 28))})
    res = await run(reader, folder_id=1)
    assert res.playlist_size == 8 and all(r.post_id >= 20 for r in res.items)
    with pytest.raises(ValueError):
        await run(reader)  # neither
    with pytest.raises(ValueError):
        await run(reader, folder_id=1, post_ids=[1])  # both


async def test_unembedded_posts_are_reported_not_fatal_and_all_missing_is_an_error(corpus):
    reader = FakeReader(corpus)
    res = await run(reader, post_ids=[0, 1, 2, 3, 999])
    assert res.playlist_size == 5 and res.embedded == 4 and res.missing_post_ids == [999]
    with pytest.raises(EmptyPlaylistError):
        await run(reader, post_ids=[998, 999])


async def test_feedback_yes_joins_playlist_no_is_hidden_unrated_is_untouched(corpus):
    """Absence from the playlist is not a 'no': only an explicit rejection excludes a story."""
    reader = FakeReader(corpus)
    base = await run(reader, post_ids=[0, 1, 2, 3])
    top = base.items[0].post_id
    second = base.items[1].post_id

    playlist = [0, 1, 2, 3]
    after = await run(reader, post_ids=playlist, confirmed_ids=[top], rejected_ids=[second])
    got = {r.post_id for r in after.items}
    assert top not in got and second not in got  # confirmed is now in the playlist; rejected is hidden
    assert after.playlist_size == 5 and after.rejected == 1 and len(got) == 5
    # The ONLY things kept out of the candidate search are the playlist, the 'yes' and the 'no'.
    # Every other story, rated or not, stays eligible: absence is not a 'no'.
    assert set(reader.excludes[-1]) == set(playlist) | {top, second}
