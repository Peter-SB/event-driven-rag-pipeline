"""Unit tests for the pure playlist-scoring maths (no DB, no network)."""
from __future__ import annotations

import numpy as np
import pytest

from event_driven_rag_service.services.playlist_scoring import (
    build_profile,
    knn_density,
    mmr_select,
    normalise,
)

D = 48


def blob(rng: np.random.Generator, centre: np.ndarray, n: int, noise: float = 0.15) -> np.ndarray:
    return normalise(centre + noise * rng.standard_normal((n, D)))


@pytest.fixture
def centres():
    rng = np.random.default_rng(1)
    return normalise(rng.standard_normal((3, D)))


def test_two_themes_are_found_and_one_theme_is_not_split(centres):
    rng = np.random.default_rng(2)
    two = np.vstack([blob(rng, centres[0], 14), blob(rng, centres[1], 10)])
    profile = build_profile(two)
    assert profile.k == 2
    assert sorted(profile.sizes) == [10, 14]
    assert profile.silhouette is not None and profile.silhouette > 0.3

    assert build_profile(blob(rng, centres[0], 24)).k == 1


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_tiny_playlists_get_a_single_theme(centres, n):
    rng = np.random.default_rng(3)
    profile = build_profile(blob(rng, centres[0], n))
    assert profile.k == 1 and profile.sizes == [n] and profile.silhouette is None


def test_profile_is_deterministic(centres):
    rng = np.random.default_rng(4)
    pl = np.vstack([blob(rng, centres[0], 12), blob(rng, centres[1], 12)])
    a, b = build_profile(pl), build_profile(pl)
    assert np.array_equal(a.labels, b.labels)


def test_heat_follows_the_playlist_not_its_average(centres):
    """A candidate in either theme is hot; one in the empty middle (where the mean sits) is not."""
    rng = np.random.default_rng(5)
    pl = np.vstack([blob(rng, centres[0], 12), blob(rng, centres[1], 12)])
    in_a = blob(rng, centres[0], 1)
    in_b = blob(rng, centres[1], 1)
    middle = normalise(centres[0] + centres[1])
    elsewhere = blob(rng, centres[2], 1)

    heat = knn_density(np.vstack([in_a, in_b, middle, elsewhere]), pl, knn=3).heat
    assert heat[0] > heat[2] and heat[1] > heat[2]
    assert heat[2] > heat[3]
    # ...whereas a single centroid would rank the middle point first
    mean = normalise(pl.mean(axis=0))[0]
    assert (middle @ mean).item() > (in_a @ mean).item()


def test_knn_reports_nearest_playlist_items_first_and_caps_at_playlist_size(centres):
    rng = np.random.default_rng(6)
    pl = blob(rng, centres[0], 2)
    d = knn_density(pl[:1], pl, knn=10)  # more neighbours requested than items
    assert d.neighbours.shape == (1, 2)
    assert d.neighbours[0, 0] == 0  # itself is nearest
    assert d.neighbour_sims[0, 0] >= d.neighbour_sims[0, 1]


def test_mmr_skips_near_duplicates_but_lambda_one_does_not(centres):
    rng = np.random.default_rng(7)
    a = blob(rng, centres[0], 1)
    cands = np.vstack([a, a + 1e-4, blob(rng, centres[1], 1)])
    heat = np.array([0.80, 0.79, 0.70])
    assert mmr_select(heat, cands, 2, mmr_lambda=0.5) == [0, 2]  # duplicate of #0 skipped
    assert mmr_select(heat, cands, 2, mmr_lambda=1.0) == [0, 1]  # pure ranking keeps it


def test_mmr_handles_k_larger_than_pool_and_empty(centres):
    rng = np.random.default_rng(8)
    cands = blob(rng, centres[0], 3)
    assert sorted(mmr_select(np.array([3.0, 2.0, 1.0]), cands, 10)) == [0, 1, 2]
    assert mmr_select(np.array([]), np.zeros((0, D)), 5) == []
