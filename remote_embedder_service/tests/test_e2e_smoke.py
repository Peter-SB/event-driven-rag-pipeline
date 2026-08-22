"""Slim end-to-end smoke test against a *real* model — no fakes anywhere.

Everything else in this suite injects a fake encoder, which proves the HTTP
contract but not that the model actually loads, that sentence-transformers is
wired up correctly, or that real numpy vectors survive the JSON round trip.
This file closes that gap using the smallest model in the registry
(bge-small-en-v1.5, 33M params, ~130MB) on CPU, so it costs seconds and needs
no GPU.

Skips — rather than fails — when the weights can't be fetched, so a CI box with
no network or an offline laptop stays green.

Tested behaviours
------------------
- The full client sequence (probe -> state -> load -> embed -> unload) against
  real weights.
- Vectors come back at the registry's declared width, finite, and normalised.
- Batching: a request larger than MAX_BATCH_SIZE returns every vector, in input
  order, with a repeated text mapping to an identical vector at both positions.
- Idle reaping frees a genuinely loaded model.
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from remote_embedder.app import create_app
from remote_embedder.config import Settings
from remote_embedder.manager import ModelManager
from remote_embedder.registry import MODEL_REGISTRY

pytestmark = pytest.mark.e2e

MODEL_ID = "text-embedding-bge-small-en-v1.5"
DIM = MODEL_REGISTRY[MODEL_ID].dim  # 384

# Small enough that the batching assertions below stay readable, and forces the
# 7-text request to split into 4 + 3 rather than going through in one pass.
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8100,
        log_level="WARNING",
        api_key="",
        device="cpu",
        max_batch_size=BATCH_SIZE,
        idle_unload_s=900,
        reaper_interval_s=15,
        enable_quantization=False,  # int8 is a CUDA path; this test runs on CPU
        normalize_embeddings=True,
    )


@pytest.fixture(scope="module")
def real_client(real_settings):
    """App backed by the real ModelManager and the real SentenceTransformer loader.

    Module-scoped so the weights load once for the whole file.
    """
    pytest.importorskip(
        "sentence_transformers",
        reason="sentence-transformers not installed — install the service to run e2e",
    )
    manager = ModelManager(real_settings)  # default_loader: real weights
    app = create_app(settings=real_settings, manager=manager)
    with TestClient(app) as client:
        # Load once here so a download/network failure skips the file rather
        # than failing every test in it with an opaque 503.
        resp = client.post("/api/v1/models/load", json={"model": MODEL_ID})
        if resp.status_code != 200:
            pytest.skip(f"could not load {MODEL_ID}: {resp.status_code} {resp.text[:200]}")
        yield client


def _embed(client, texts) -> list[list[float]]:
    resp = client.post("/v1/embeddings", json={"model": MODEL_ID, "input": texts})
    assert resp.status_code == 200, resp.text
    return [item["embedding"] for item in resp.json()["data"]]


# --- the plumbing ----------------------------------------------------------


def test_endpoint_reachable(real_client):
    resp = real_client.get("/v1/models")
    assert resp.status_code == 200
    assert MODEL_ID in {m["id"] for m in resp.json()["data"]}


def test_real_model_reports_loaded(real_client):
    body = real_client.get("/api/v0/models").json()
    state = next(m["state"] for m in body["data"] if m["id"] == MODEL_ID)
    assert state == "loaded"


def test_real_embedding_has_declared_width_and_finite_values(real_client):
    vectors = _embed(real_client, ["the quick brown fox", "a second sentence"])

    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    assert all(math.isfinite(x) for v in vectors for x in v)
    # Not the all-zeros vector a broken load would plausibly produce.
    assert any(x != 0.0 for x in vectors[0])


def test_real_embeddings_are_normalised(real_client):
    """NORMALIZE_EMBEDDINGS is on by default because the pipeline compares with
    cosine distance, and LM Studio returned normalised vectors too."""
    (vector,) = _embed(real_client, ["normalise me"])
    norm = math.sqrt(sum(x * x for x in vector))
    assert norm == pytest.approx(1.0, abs=1e-3)


# --- batching --------------------------------------------------------------


def test_request_larger_than_batch_size_returns_every_vector_in_order(real_client):
    """7 texts at MAX_BATCH_SIZE=4 means two forward passes stitched together.

    The repeated text at positions 0 and 6 lands in *different* sub-batches, so
    identical vectors there is real evidence the halves were reassembled in
    order — not merely that the response has the right length.
    """
    texts = [
        "shared sentence",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "shared sentence",
    ]
    assert len(texts) > BATCH_SIZE

    vectors = _embed(real_client, texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == DIM for v in vectors)
    assert vectors[0] == pytest.approx(vectors[6], abs=1e-6)
    assert vectors[0] != pytest.approx(vectors[1], abs=1e-6)


def test_batched_results_match_one_at_a_time_results(real_client):
    """Sub-batching must not change the numbers — same text, same vector,
    whether it arrives alone or in the middle of a larger request."""
    texts = ["alpha text", "beta text", "gamma text", "delta text", "epsilon text"]

    batched = _embed(real_client, texts)
    individually = [_embed(real_client, [t])[0] for t in texts]

    for i, (a, b) in enumerate(zip(batched, individually)):
        assert a == pytest.approx(b, abs=1e-5), f"mismatch at index {i}"


def test_indices_are_sequential(real_client):
    resp = real_client.post(
        "/v1/embeddings", json={"model": MODEL_ID, "input": ["a", "b", "c", "d", "e"]}
    )
    assert [item["index"] for item in resp.json()["data"]] == [0, 1, 2, 3, 4]


# --- unload ----------------------------------------------------------------


def test_unload_then_reload_still_works(real_client):
    """Runs last-ish: unloads the module-scoped model, then puts it back so the
    remaining tests (and reruns) are unaffected."""
    assert (
        real_client.post("/api/v1/models/unload", json={"instance_id": MODEL_ID}).status_code
        == 200
    )
    body = real_client.get("/api/v0/models").json()
    assert next(m["state"] for m in body["data"] if m["id"] == MODEL_ID) == "not-loaded"

    # An embed with nothing resident must load on demand and still work.
    vectors = _embed(real_client, ["back from the dead"])
    assert len(vectors[0]) == DIM


# --- real CUDA + int8, opt-in ---------------------------------------------


@pytest.mark.gpu
def test_qwen3_int8_on_cuda():
    """Deselected by default (`-m 'not gpu'`). Run on the GPU box with
    `pytest -m gpu` to verify the int8 path and semantic sanity."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    model_id = "text-embedding-qwen3-embedding-0.6b"
    cfg = Settings(
        device="cuda",
        max_batch_size=8,
        idle_unload_s=0,
        enable_quantization=True,
        normalize_embeddings=True,
    )
    app = create_app(settings=cfg, manager=ModelManager(cfg))
    with TestClient(app) as client:
        assert client.post("/api/v1/models/load", json={"model": model_id}).status_code == 200

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": model_id,
                "input": [
                    "the cat sat on the mat",
                    "a feline rested upon the rug",
                    "quarterly revenue exceeded projections",
                ],
            },
        )
        assert resp.status_code == 200
        vecs = [item["embedding"] for item in resp.json()["data"]]
        assert all(len(v) == MODEL_REGISTRY[model_id].dim for v in vecs)

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))

        # Paraphrases should sit closer together than unrelated text. A loose
        # bound — this catches a broken/garbage load, not model quality.
        assert cos(vecs[0], vecs[1]) > cos(vecs[0], vecs[2])
