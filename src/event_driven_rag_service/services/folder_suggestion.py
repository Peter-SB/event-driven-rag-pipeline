"""Folder suggestion — orchestrates the read-only repository and the scoring maths.

Flow: load folder vectors -> cluster into themes -> one ANN query per theme
centroid (candidate generation) -> kNN-density "heat" for every candidate ->
MMR to pick ``k`` diverse results. See ``folder_scoring`` for the why.

Depends only on the ``SuggestionReader`` protocol so it unit-tests with an
in-memory fake.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from event_driven_rag_service.repository.folder_suggestion_repository import CandidateRow
from event_driven_rag_service.services.folder_scoring import (
    avoid_penalty,
    build_profile,
    knn_density,
    mmr_select,
    normalise,
)

# Candidates fetched per theme centroid = max(k * POOL_FACTOR, MIN_POOL).
POOL_FACTOR = 5

# How much a "very yes" folder item counts in the neighbour average (a plain item counts 1).
VERY_YES_WEIGHT = 2.0
# Candidates close to a "very no" story lose up to this much heat (heat is a cosine similarity).
VERY_NO_PENALTY = 0.5
MIN_POOL = 40


class EmptyFolderError(Exception):
    """The folder has no posts with an embedding to learn from."""


class SuggestionReader(Protocol):
    async def folder_post_ids(self, posts_table: str, folder_id: int) -> list[int]: ...

    async def folder_embeddings(
        self, chunks_table: str, post_ids: list[int]
    ) -> dict[int, np.ndarray]: ...

    async def nearest_candidates(
        self,
        chunks_table: str,
        posts_table: str,
        query: np.ndarray,
        limit: int,
        exclude_post_ids: list[int],
    ) -> list[CandidateRow]: ...

    async def post_titles(self, posts_table: str, post_ids: list[int]) -> dict[int, str]: ...


@dataclass(frozen=True)
class SuggestParams:
    k: int = 10
    knn: int = 3
    max_clusters: int = 5
    diversity: float = 0.3  # 0 = pure heat ranking, 1 = maximum variety (MMR lambda = 1 - diversity)


@dataclass(frozen=True)
class Because:
    post_id: int
    title: str
    similarity: float


@dataclass(frozen=True)
class Suggestion:
    rank: int
    post_id: int
    title: str
    subreddit: str | None
    url: str | None
    score: float          # kNN-density heat: mean cosine similarity to nearest folder items
    theme: int            # index into SuggestResult.themes
    because: list[Because]


@dataclass(frozen=True)
class Theme:
    index: int
    size: int
    exemplars: list[Because]  # folder items closest to the theme centre (similarity = to centre)


@dataclass(frozen=True)
class SuggestResult:
    folder_size: int
    embedded: int
    missing_post_ids: list[int]
    silhouette: float | None
    candidates_considered: int
    rejected: int = 0
    themes: list[Theme] = field(default_factory=list)
    items: list[Suggestion] = field(default_factory=list)


async def suggest_for_folder(
    reader: SuggestionReader,
    *,
    posts_table: str,
    chunks_table: str,
    post_ids: list[int] | None = None,
    folder_id: int | None = None,
    verdicts: Mapping[int, str] | None = None,
    params: SuggestParams = SuggestParams(),
) -> SuggestResult:
    if (post_ids is None) == (folder_id is None):
        raise ValueError("give exactly one of post_ids or folder_id")
    if folder_id is not None:
        post_ids = await reader.folder_post_ids(posts_table, folder_id)
    # Explicit feedback (very_no / no / yes / very_yes). A yes joins the folder as a
    # positive, and a very_yes counts double in the neighbour average. A no is hidden
    # from the results; a very_no is hidden and also demotes its lookalikes. Unrated
    # stories are unlabelled, never negatives.
    verdicts = verdicts or {}
    rejected = sorted(p for p, v in verdicts.items() if v in ("no", "very_no"))
    very_no = sorted(p for p, v in verdicts.items() if v == "very_no")
    ids = sorted(set(post_ids or []) | {p for p, v in verdicts.items() if v in ("yes", "very_yes")})

    vectors = await reader.folder_embeddings(chunks_table, ids) if ids else {}
    if not vectors:
        raise EmptyFolderError(
            f"none of the {len(ids)} folder posts has an embedding in {chunks_table!r}"
        )
    embedded_ids = sorted(vectors)
    folder = normalise(np.stack([vectors[i] for i in embedded_ids]))
    missing = [i for i in ids if i not in vectors]

    profile = build_profile(folder, max_clusters=params.max_clusters)

    # --- candidate generation: one ANN query per theme centroid -------------------
    pool = max(params.k * POOL_FACTOR, MIN_POOL)
    exclude = sorted(set(ids) | set(rejected))
    by_post: dict[int, CandidateRow] = {}
    for centroid in profile.centroids:
        for row in await reader.nearest_candidates(chunks_table, posts_table, centroid, pool, exclude):
            by_post.setdefault(row.post_id, row)
    candidates = list(by_post.values())

    titles = await reader.post_titles(posts_table, embedded_ids)

    # Theme summaries: the folder items closest to each centre.
    themes: list[Theme] = []
    for c in range(profile.k):
        members = np.flatnonzero(profile.labels == c)
        sims = folder[members] @ profile.centroids[c]
        order = members[np.argsort(-sims)][:3]
        themes.append(
            Theme(
                index=c,
                size=int(len(members)),
                exemplars=[
                    Because(embedded_ids[m], titles.get(embedded_ids[m], ""), float(folder[m] @ profile.centroids[c]))
                    for m in order
                ],
            )
        )

    items: list[Suggestion] = []
    if candidates:
        cand_matrix = normalise(np.stack([c.embedding for c in candidates]))
        weights = np.array(
            [VERY_YES_WEIGHT if verdicts.get(i) == "very_yes" else 1.0 for i in embedded_ids],
            dtype=np.float32,
        )
        density = knn_density(cand_matrix, folder, params.knn, weights)
        avoid_vectors = await reader.folder_embeddings(chunks_table, very_no) if very_no else {}
        avoid = np.stack(list(avoid_vectors.values())) if avoid_vectors else np.empty((0, folder.shape[1]))
        ranking_heat = density.heat - avoid_penalty(cand_matrix, avoid, VERY_NO_PENALTY)
        picked = mmr_select(ranking_heat, cand_matrix, params.k, 1.0 - params.diversity)
        for rank, idx in enumerate(picked, start=1):
            cand = candidates[idx]
            nearest = density.neighbours[idx]
            items.append(
                Suggestion(
                    rank=rank,
                    post_id=cand.post_id,
                    title=cand.title,
                    subreddit=cand.subreddit,
                    url=cand.url,
                    score=float(density.heat[idx]),
                    theme=int(profile.labels[nearest[0]]),
                    because=[
                        Because(embedded_ids[n], titles.get(embedded_ids[n], ""), float(s))
                        for n, s in zip(nearest[:2], density.neighbour_sims[idx][:2])
                    ],
                )
            )

    return SuggestResult(
        folder_size=len(ids),
        embedded=len(embedded_ids),
        missing_post_ids=missing,
        silhouette=profile.silhouette,
        candidates_considered=len(candidates),
        rejected=len(rejected),
        themes=themes,
        items=items,
    )
