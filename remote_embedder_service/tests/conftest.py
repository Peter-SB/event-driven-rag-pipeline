"""Shared fixtures.

The default fixtures build the app around a *fake* loader, so the whole contract
suite runs on any machine with no GPU and no model download. Only
``test_e2e_smoke.py`` opts into real weights.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from remote_embedder.app import create_app
from remote_embedder.config import Settings
from remote_embedder.manager import ModelManager
from remote_embedder.registry import ModelSpec

BGE_BASE = "text-embedding-bge-base-en-v1.5"
BGE_SMALL = "text-embedding-bge-small-en-v1.5"
QWEN = "text-embedding-qwen3-embedding-0.6b"


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.

    Produces a `dim`-wide vector whose leading values are derived from the text,
    so tests can assert that a given input maps to a stable vector and that
    ordering survives sub-batching.
    """

    def __init__(self, dim: int, spec_id: str = "") -> None:
        self.dim = dim
        self.spec_id = spec_id
        self.calls: list[list[str]] = []  # one entry per encode() = per sub-batch

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        seed = float(sum(ord(c) for c in text) % 997) / 997.0
        return [seed] + [0.0] * (self.dim - 1)


class FakeReranker:
    """Deterministic stand-in for the Qwen3-Reranker causal LM.

    Scores by (bounded) lexical overlap between query and document — enough
    to exercise ordering and the input-alignment contract without loading a
    real model.
    """

    def __init__(self, spec_id: str = "") -> None:
        self.spec_id = spec_id
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        query_terms = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(overlap / max(len(query_terms), 1))
        return scores


class RecordingLoader:
    """Loader that hands out FakeEncoders/FakeRerankers and remembers what it
    was asked for. Dispatches by `spec.kind`, mirroring
    `remote_embedder.manager.default_loader`."""

    def __init__(self, dim_override: int | None = None) -> None:
        self.loaded: list[str] = []
        self.dim_override = dim_override

    def __call__(self, spec: ModelSpec, settings: Settings) -> FakeEncoder | FakeReranker:
        self.loaded.append(spec.id)
        if spec.kind == "rerank":
            return FakeReranker(spec.id)
        return FakeEncoder(self.dim_override or spec.dim, spec.id)


class FakeClock:
    """Manually advanced monotonic clock for the idle-reaper tests."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def settings() -> Settings:
    """Settings with the environment ignored, so a developer's own .env or
    exported DEVICE/API_KEY can't change what these tests assert.

    log_level defaults to WARNING here to keep test output quiet — tests that
    exercise logging behaviour at INFO (test_logging_config.py) override it.
    """
    return Settings(
        host="127.0.0.1",
        port=8100,
        log_level="WARNING",
        api_key="",
        device="cpu",
        max_batch_size=32,
        idle_unload_s=900,
        reaper_interval_s=15,
        enable_quantization=False,
        normalize_embeddings=True,
    )


@pytest.fixture
def loader() -> RecordingLoader:
    return RecordingLoader()


@pytest.fixture
def manager(settings: Settings, loader: RecordingLoader) -> ModelManager:
    return ModelManager(settings, loader=loader)


@pytest.fixture
def client(settings: Settings, manager: ModelManager):
    """TestClient over the real app wired to the fake-backed manager.

    The reaper thread would start on lifespan entry; idle_unload_s is long
    enough that it never fires during a test, and reaper behaviour is covered
    deterministically in test_manager.py instead.
    """
    app = create_app(settings=settings, manager=manager)
    with TestClient(app) as c:
        yield c
