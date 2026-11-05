"""Pure maths for folder suggestion — numpy only, no I/O.

The idea
--------
A folder is a *set* of embedding vectors, usually with several themes, so a
single average vector (centroid) lands in the empty space between themes. We
instead:

1. **Cluster** the folder (spherical k-means, k picked by silhouette) — used
   only to *generate candidates*: one ANN query per cluster centroid.
2. **Score** each candidate with a *kNN density estimate*: the mean cosine
   similarity to its ``knn`` nearest folder items. This is the "heat" at the
   candidate's location. We deliberately do not fit a real density (KDE / GMM
   likelihood) — in 384–2560 dimensions distances concentrate and bandwidths are
   meaningless, whereas a nearest-neighbour similarity stays well behaved and
   handles multi-theme folders with no clustering assumptions.
3. **Diversify** with MMR so the top results are not ten near-duplicates.

Everything here works on L2-normalised vectors so a dot product is the cosine
similarity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows (zero rows stay zero). Returns float32."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FolderProfile:
    """The folder's themes. ``centroids`` are unit vectors, one per cluster."""

    centroids: np.ndarray        # (k, D)
    labels: np.ndarray           # (P,) cluster index of each folder item
    sizes: list[int]             # items per cluster
    silhouette: float | None     # None when k == 1 (undefined)

    @property
    def k(self) -> int:
        return len(self.sizes)


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = [int(rng.integers(len(x)))]
    best_sim = x @ x[idx[0]]
    for _ in range(1, k):
        dist = np.clip(1.0 - best_sim, 0.0, None) ** 2
        total = dist.sum()
        if total <= 0:  # all remaining points coincide with a chosen centre
            nxt = int(rng.integers(len(x)))
        else:
            nxt = int(rng.choice(len(x), p=dist / total))
        idx.append(nxt)
        best_sim = np.maximum(best_sim, x @ x[nxt])
    return x[idx].copy()


def spherical_kmeans(
    x: np.ndarray, k: int, *, seed: int = 0, n_init: int = 5, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """K-means on the unit sphere (cosine). Returns ``(labels, centroids)`` of the best run."""
    x = normalise(x)
    k = min(k, len(x))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centroids = _kmeans_plus_plus(x, k, rng)
        labels = np.full(len(x), -1)
        for _ in range(max_iter):
            sims = x @ centroids.T
            new_labels = sims.argmax(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if len(members) == 0:  # re-seed an empty cluster at the worst-fit point
                    centroids[c] = x[int(sims.max(axis=1).argmin())]
                else:
                    centroids[c] = normalise(members.sum(axis=0))[0]
        score = float((x @ centroids.T).max(axis=1).sum())  # total within-cluster similarity
        if best is None or score > best[0]:
            best = (score, labels.copy(), centroids.copy())
    assert best is not None
    return best[1], best[2]


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette using cosine distance (1 - similarity). Range [-1, 1]."""
    x = normalise(x)
    dist = np.clip(1.0 - x @ x.T, 0.0, None)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    scores = np.zeros(len(x))
    for i in range(len(x)):
        own = labels == labels[i]
        n_own = own.sum()
        if n_own <= 1:
            continue  # singleton cluster scores 0
        a = dist[i, own].sum() / (n_own - 1)
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        scores[i] = (b - a) / max(a, b, 1e-12)
    return float(scores.mean())


def build_profile(
    folder: np.ndarray,
    *,
    max_clusters: int = 5,
    min_cluster_size: int = 4,
    min_silhouette: float = 0.15,
    seed: int = 0,
) -> FolderProfile:
    """Split the folder into themes, or keep it as one when it has no clear structure.

    Tries k = 2 .. min(max_clusters, P // min_cluster_size) and keeps the k with
    the best silhouette, provided no cluster is a singleton and the silhouette
    clears ``min_silhouette``. Otherwise falls back to a single centroid.

    The 0.15 floor is empirical: a single Gaussian blob scores ~0.10 under k-means
    and real folders in the homelab library scored 0.06-0.13, i.e. noise-level
    "themes". Clustering only feeds candidate generation (and the UI's theme chips);
    ranking is by kNN density, so a one-theme fallback costs no recall (measured:
    recall@30 0.109 with 1 cluster vs 0.113-0.116 with 3-5).
    """
    x = normalise(folder)
    n = len(x)
    one = FolderProfile(
        centroids=normalise(x.sum(axis=0)),
        labels=np.zeros(n, dtype=int),
        sizes=[n],
        silhouette=None,
    )
    k_max = min(max_clusters, n // min_cluster_size)
    if k_max < 2:
        return one

    best: tuple[float, FolderProfile] | None = None
    for k in range(2, k_max + 1):
        labels, centroids = spherical_kmeans(x, k, seed=seed)
        sizes = [int((labels == c).sum()) for c in range(k)]
        if min(sizes) < 2:
            continue
        s = silhouette(x, labels)
        if best is None or s > best[0]:
            best = (s, FolderProfile(centroids=centroids, labels=labels, sizes=sizes, silhouette=s))
    if best is None or best[0] < min_silhouette:
        return one
    return best[1]


# ---------------------------------------------------------------------------
# Scoring and selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityScores:
    heat: np.ndarray          # (C,) mean similarity to the knn nearest folder items
    neighbours: np.ndarray    # (C, m) folder indices of those neighbours, nearest first
    neighbour_sims: np.ndarray  # (C, m)


def knn_density(
    candidates: np.ndarray,
    folder: np.ndarray,
    knn: int = 3,
    weights: np.ndarray | None = None,
) -> DensityScores:
    """kNN density estimate ("heat") of each candidate within the folder.

    ``weights`` (one per folder item, default all 1) makes some items count for more in
    the neighbour average: a "very yes" story pulls its lookalikes up harder.
    """
    c = normalise(candidates)
    p = normalise(folder)
    m = max(1, min(knn, len(p)))
    sims = c @ p.T
    top = np.argpartition(-sims, m - 1, axis=1)[:, :m]
    top_sims = np.take_along_axis(sims, top, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top = np.take_along_axis(top, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    if weights is None:
        heat = top_sims.mean(axis=1)
    else:
        w = np.asarray(weights, dtype=np.float32)[top]
        heat = (top_sims * w).sum(axis=1) / w.sum(axis=1)
    return DensityScores(heat=heat, neighbours=top, neighbour_sims=top_sims)


def avoid_penalty(candidates: np.ndarray, avoid: np.ndarray, strength: float) -> np.ndarray:
    """Per-candidate penalty for being close to stories the user said "very no" to.

    ``strength`` x the highest cosine similarity to any avoided story (never negative),
    so lookalikes of a rejected story rank below equally hot candidates that are not.
    """
    if len(avoid) == 0:
        return np.zeros(len(candidates), dtype=np.float32)
    sims = normalise(candidates) @ normalise(avoid).T
    return strength * np.clip(sims.max(axis=1), 0.0, None)


def mmr_select(
    heat: np.ndarray, candidates: np.ndarray, k: int, mmr_lambda: float = 0.7
) -> list[int]:
    """Maximal Marginal Relevance: balance heat against similarity to already-picked items.

    ``mmr_lambda`` = 1 ranks purely by heat; lower values push harder for variety.
    Returns candidate indices in selection order.
    """
    n = len(heat)
    k = min(k, n)
    if k <= 0:
        return []
    c = normalise(candidates)
    selected: list[int] = [int(np.argmax(heat))]
    max_sim_to_selected = c @ c[selected[0]]
    chosen = np.zeros(n, dtype=bool)
    chosen[selected[0]] = True
    while len(selected) < k:
        mmr = mmr_lambda * heat - (1.0 - mmr_lambda) * max_sim_to_selected
        mmr[chosen] = -np.inf
        nxt = int(np.argmax(mmr))
        selected.append(nxt)
        chosen[nxt] = True
        max_sim_to_selected = np.maximum(max_sim_to_selected, c @ c[nxt])
    return selected
