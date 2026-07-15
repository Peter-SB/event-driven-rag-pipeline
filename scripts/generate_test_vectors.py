#!/usr/bin/env python3
"""Generate and verify pre-computed test vectors for /search/similar tests.

Prints the vectors used in integration and e2e tests so they can be verified
independently of the test runner.  All test cases use unit vectors at
orthogonal axes so cosine similarities are exact and predictable:

    source chunks (body): unit axes 0, 1, 2
    averaged query:       [1/3, 1/3, 1/3, 0, ...]
    close neighbour:      normalised [1, 1, 1, 0, ...] → cosine_sim = 1.0
    far neighbour:        unit axis 3 → cosine_sim = 0.0

Usage:
    python scripts/generate_test_vectors.py
"""
from __future__ import annotations

import math

EMBED_DIMS: dict[str, int] = {
    "body": 768,            # BAAI/bge-base-en-v1.5
    "title": 384,           # BAAI/bge-small-en-v1.5
    "summary_title": 1024,  # Qwen/Qwen3-0.6B
}


def unit_vec(dim: int, axis: int) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def normalized(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def average(vecs: list[list[float]]) -> list[float]:
    n, dim = len(vecs), len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _fmt(v: list[float], n: int = 5) -> str:
    return str([round(x, 6) for x in v[:n]])


if __name__ == "__main__":
    for chunk_type, dim in EMBED_DIMS.items():
        u0 = unit_vec(dim, 0)
        u1 = unit_vec(dim, 1)
        u2 = unit_vec(dim, 2)
        u3 = unit_vec(dim, 3)

        avg = average([u0, u1, u2])          # body: average of 3 source chunks
        close = normalized([1.0, 1.0, 1.0] + [0.0] * (dim - 3))
        far = u3

        print(f"\n{'='*60}")
        print(f"  {chunk_type}  (dim={dim})")
        print(f"{'='*60}")
        print(f"  source_chunk_0 (axis 0) [:5]: {_fmt(u0)}")
        print(f"  source_chunk_1 (axis 1) [:5]: {_fmt(u1)}")
        print(f"  source_chunk_2 (axis 2) [:5]: {_fmt(u2)}")
        print(f"  body average            [:5]: {_fmt(avg)}")
        print(f"  close_neighbour         [:5]: {_fmt(close)}")
        print(f"  far_neighbour  (axis 3) [:5]: {_fmt(far)}")
        print(f"  cosine(avg,  close): {cosine_sim(avg,  close):.6f}  (expect 1.000000)")
        print(f"  cosine(avg,  far):   {cosine_sim(avg,  far):.6f}  (expect 0.000000)")
        print(f"  cosine(u0,   close): {cosine_sim(u0,   close):.6f}  (title/summary query)")
        print(f"  cosine(u0,   u1):    {cosine_sim(u0,   u1):.6f}  (orthogonal sanity check)")
