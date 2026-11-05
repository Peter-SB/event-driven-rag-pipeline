"""Opt-in checks of folder suggestion against a *real* library database.

Run with the homelab Postgres (strictly read-only: the pool sets
``default_transaction_read_only=on`` so any accidental write errors out):

    SUGGEST_LIVE_DB_URL=postgresql://rag:rag@192.168.0.234:5432/rag \
    SUGGEST_LIVE_LIBRARY=reddit_posts_main SUGGEST_LIVE_FOLDER=1 \
        pytest -m live_db tests/integration/test_folder_suggestion_live.py -s

Skipped unless SUGGEST_LIVE_DB_URL is set. Not part of the default run.

What it checks: hold out part of a folder, recommend from the rest, and measure how
many held-out stories come back. That recall is a *lower bound* on quality: a
recommended story that is not in the folder is not necessarily a miss, since
absence from a folder is not a "no" (that is what the UI's yes/no is for).
"""
from __future__ import annotations

import os
import random

import asyncpg
import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.repository.folder_suggestion_repository import FolderSuggestionRepository
from event_driven_rag_service.services.folder_suggestion import SuggestParams, suggest_for_folder
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name

DB_URL = os.environ.get("SUGGEST_LIVE_DB_URL")
LIBRARY = os.environ.get("SUGGEST_LIVE_LIBRARY", "reddit_posts_main")
FOLDERS = [int(x) for x in os.environ.get("SUGGEST_LIVE_FOLDER", "1,2,13").split(",")]
K = 30

pytestmark = [
    pytest.mark.live_db,
    pytest.mark.skipif(not DB_URL, reason="SUGGEST_LIVE_DB_URL not set"),
]

POSTS = f"posts_{LIBRARY}"
CHUNKS = build_chunk_table_name(POSTS, "summary_title", EMBED_CONFIGS["summary_title"].primary)


@pytest.fixture
async def repo():
    pool = await asyncpg.create_pool(
        DB_URL, min_size=1, max_size=2, server_settings={"default_transaction_read_only": "on"}
    )
    yield FolderSuggestionRepository(pool), pool
    await pool.close()


async def test_database_really_is_read_only(repo):
    _, pool = repo
    with pytest.raises(asyncpg.exceptions.ReadOnlySQLTransactionError):
        await pool.execute("CREATE TEMP TABLE should_fail (x int)")


@pytest.mark.parametrize("folder", FOLDERS)
async def test_heldout_recall_beats_random_by_a_wide_margin(repo, folder):
    r, pool = repo
    corpus = await pool.fetchval(
        f"SELECT count(*) FROM {POSTS} p WHERE NOT COALESCE(is_deleted, FALSE) AND EXISTS "
        f"(SELECT 1 FROM {CHUNKS} c WHERE c.post_id = p.post_id AND c.embedding IS NOT NULL)"
    )
    vectors = await r.folder_embeddings(CHUNKS, await r.folder_post_ids(POSTS, folder))
    ids = sorted(vectors)
    assert len(ids) >= 12, "folder too small for a hold-out test"

    rng = random.Random(folder)
    hits = expected_random = 0.0
    trials = 8
    for _ in range(trials):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = len(ids) // 4
        held, train = set(shuffled[:cut]), shuffled[cut:]
        res = await suggest_for_folder(
            r, posts_table=POSTS, chunks_table=CHUNKS, post_ids=train, params=SuggestParams(k=K)
        )
        got = [i.post_id for i in res.items]
        assert len(got) == K and len(set(got)) == K and not set(got) & set(train)
        assert all(a.score >= 0 for a in res.items)
        hits += len(set(got) & held)
        expected_random += K * cut / corpus
    lift = hits / expected_random
    print(f"\nfolder {folder}: hits={hits:.0f} random={expected_random:.1f} lift=x{lift:.1f} (corpus {corpus})")
    assert lift > 3.0


async def test_folder_suggestionations_are_hot_and_ordered_sensibly(repo):
    r, _ = repo
    res = await suggest_for_folder(
        r, posts_table=POSTS, chunks_table=CHUNKS, folder_id=FOLDERS[0], params=SuggestParams(k=20)
    )
    assert res.embedded >= 10 and res.themes and sum(t.size for t in res.themes) == res.embedded
    assert len(res.items) == 20
    folder_ids = set(await r.folder_post_ids(POSTS, FOLDERS[0]))
    assert not {i.post_id for i in res.items} & folder_ids
    assert all(i.because for i in res.items)
