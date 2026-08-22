"""The wire contract with the pipeline's FallbackEmbeddingModel.

Every assertion here mirrors something
``src/event_driven_rag_service/worker/remote_embedding.py`` actually reads. If a
test in this file fails, the LM Studio swap is not transparent and the GPU worker
will quietly fall back to local embedding instead of erroring — which is exactly
the failure mode that is hard to notice in production.

Tested behaviours
------------------
- The four-step pre-flight sequence the client runs, in order, end to end.
- GET /v1/models lists every registry id and needs no model loaded.
- GET /api/v0/models lists unloaded models with state "not-loaded" rather than
  omitting them.
- POST /api/v1/models/load flips state to "loaded".
- POST /v1/embeddings returns one embedding per input, positionally aligned.
- POST /api/v1/models/unload accepts `instance_id` and returns to "not-loaded".
- Unknown model ids and empty input are 4xx, never 500.
- `input` accepts a bare string as well as a list.
- Bearer tokens are ignored when no API_KEY is set, enforced when one is.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from remote_embedder.app import create_app
from remote_embedder.registry import MODEL_REGISTRY

BGE_BASE = "text-embedding-bge-base-en-v1.5"
QWEN = "text-embedding-qwen3-embedding-0.6b"


def _state_of(client, model_id: str) -> str | None:
    body = client.get("/api/v0/models").json()
    return next((m["state"] for m in body["data"] if m["id"] == model_id), None)


# --- the full client sequence ---------------------------------------------


def test_full_client_preflight_sequence(client):
    """Walk the exact four steps FallbackEmbeddingModel.encode() performs.

    Kept as one test on purpose: the individual steps are asserted below, but
    the *sequence* is the thing that has to work, and a per-step test can pass
    while the ordering assumption (load must precede a successful embed) is
    broken.
    """
    # 1. reachability probe
    assert client.get("/v1/models").status_code == 200

    # 2. model known to the server, but not yet resident
    assert _state_of(client, QWEN) == "not-loaded"

    # 3. request a load
    load = client.post("/api/v1/models/load", json={"model": QWEN})
    assert load.status_code == 200
    assert _state_of(client, QWEN) == "loaded"

    # 4. embed
    resp = client.post("/v1/embeddings", json={"model": QWEN, "input": ["a", "b"]})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2
    assert len(data[0]["embedding"]) == MODEL_REGISTRY[QWEN].dim


# --- GET /v1/models --------------------------------------------------------


def test_v1_models_lists_every_registry_entry(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert {m["id"] for m in body["data"]} == set(MODEL_REGISTRY)
    assert all(m["object"] == "model" for m in body["data"])


def test_v1_models_does_not_load_anything(client, loader):
    """The client fires this before *every* batch — it must stay free."""
    client.get("/v1/models")
    assert loader.loaded == []


# --- GET /api/v0/models ----------------------------------------------------


def test_unloaded_models_are_listed_with_not_loaded_state(client):
    """Regression: unloaded models must be *present* with state "not-loaded".

    The client distinguishes "endpoint is down" from "endpoint doesn't have this
    model" by whether the id appears in this listing. If we omitted unloaded
    models, `is_model_available()` would return False and the batch would go to
    the local fallback — the load path would never be reached and the remote
    service would sit idle forever.
    """
    body = client.get("/api/v0/models").json()
    ids = {m["id"] for m in body["data"]}
    assert ids == set(MODEL_REGISTRY)
    assert all(m["state"] == "not-loaded" for m in body["data"])


def test_api_v0_models_exposes_state_field_the_client_reads(client):
    """The client keys off `state`, not presence. Guard the field name."""
    entry = client.get("/api/v0/models").json()["data"][0]
    assert "state" in entry
    assert entry["state"] in {"loaded", "not-loaded"}


def test_only_one_model_is_loaded_at_a_time(client):
    client.post("/api/v1/models/load", json={"model": QWEN})
    client.post("/api/v1/models/load", json={"model": BGE_BASE})

    body = client.get("/api/v0/models").json()
    loaded = [m["id"] for m in body["data"] if m["state"] == "loaded"]
    assert loaded == [BGE_BASE]


# --- POST /api/v1/models/load ---------------------------------------------


def test_load_is_idempotent(client, loader):
    client.post("/api/v1/models/load", json={"model": QWEN})
    client.post("/api/v1/models/load", json={"model": QWEN})
    assert loader.loaded == [QWEN]


def test_load_unknown_model_is_404_not_500(client):
    resp = client.post("/api/v1/models/load", json={"model": "no-such-model"})
    assert resp.status_code == 404


# --- POST /v1/embeddings ---------------------------------------------------


def test_embeddings_loads_on_demand_without_explicit_load(client):
    """The client only calls /load when state says not-loaded; a race or a
    restart could still send an embed first. It must just work."""
    resp = client.post("/v1/embeddings", json={"model": BGE_BASE, "input": ["x"]})
    assert resp.status_code == 200
    assert len(resp.json()["data"][0]["embedding"]) == MODEL_REGISTRY[BGE_BASE].dim


def test_embeddings_preserve_input_order(client):
    """The client zips `data` against its own input list positionally, so a
    reordered response would silently attach every embedding to the wrong chunk."""
    texts = ["alpha", "beta", "gamma", "delta"]
    resp = client.post("/v1/embeddings", json={"model": BGE_BASE, "input": texts})
    data = resp.json()["data"]

    assert [item["index"] for item in data] == [0, 1, 2, 3]

    # Re-embed each text alone; each must match its position in the batch.
    for i, text in enumerate(texts):
        solo = client.post(
            "/v1/embeddings", json={"model": BGE_BASE, "input": [text]}
        ).json()["data"][0]["embedding"]
        assert data[i]["embedding"] == solo


def test_embeddings_accepts_bare_string_input(client):
    resp = client.post("/v1/embeddings", json={"model": BGE_BASE, "input": "solo"})
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1


def test_embeddings_response_has_openai_envelope(client):
    body = client.post(
        "/v1/embeddings", json={"model": BGE_BASE, "input": ["x"]}
    ).json()
    assert body["object"] == "list"
    assert body["model"] == BGE_BASE
    assert body["data"][0]["object"] == "embedding"
    assert "usage" in body


def test_embeddings_unknown_model_is_400_not_500(client):
    resp = client.post("/v1/embeddings", json={"model": "no-such-model", "input": ["x"]})
    assert resp.status_code == 400


def test_embeddings_empty_input_is_400(client):
    resp = client.post("/v1/embeddings", json={"model": BGE_BASE, "input": []})
    assert resp.status_code == 400


# --- POST /api/v1/models/unload -------------------------------------------


def test_unload_accepts_instance_id(client):
    """The pipeline's live test unloads with {"instance_id": ...} to force the
    not-loaded precondition. That exact spelling has to work."""
    client.post("/api/v1/models/load", json={"model": QWEN})
    assert _state_of(client, QWEN) == "loaded"

    resp = client.post("/api/v1/models/unload", json={"instance_id": QWEN})
    assert resp.status_code == 200
    assert _state_of(client, QWEN) == "not-loaded"


def test_unload_accepts_model_alias(client):
    client.post("/api/v1/models/load", json={"model": QWEN})
    resp = client.post("/api/v1/models/unload", json={"model": QWEN})
    assert resp.status_code == 200
    assert _state_of(client, QWEN) == "not-loaded"


def test_unload_when_nothing_loaded_is_still_2xx(client):
    """The live test skips on 404/501 but otherwise expects success; unloading
    an already-unloaded model is a no-op, not an error."""
    resp = client.post("/api/v1/models/unload", json={"instance_id": QWEN})
    assert resp.status_code == 200
    assert resp.json()["state"] == "not-loaded"


# --- health ----------------------------------------------------------------


def test_health_endpoint(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["loaded_model"] is None


# --- auth ------------------------------------------------------------------


def test_bearer_token_ignored_when_no_api_key_configured(client):
    """EMBED_REMOTE_API_KEY may be set on the client side against an open
    service; the header must not cause a 401."""
    resp = client.get("/v1/models", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 200


def test_bearer_token_enforced_when_api_key_configured(settings, manager):
    settings.api_key = "s3cret"
    app = create_app(settings=settings, manager=manager)
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = c.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


def test_api_key_protects_embeddings_and_load(settings, manager):
    settings.api_key = "s3cret"
    app = create_app(settings=settings, manager=manager)
    with TestClient(app) as c:
        assert c.post("/v1/embeddings", json={"model": BGE_BASE, "input": ["x"]}).status_code == 401
        assert c.post("/api/v1/models/load", json={"model": BGE_BASE}).status_code == 401
        assert c.get("/api/v0/models").status_code == 401


# --- origin sharing --------------------------------------------------------


def test_v1_and_api_share_one_origin(client):
    """The client derives the native API base by stripping the path off
    EMBED_REMOTE_URL, so /v1/* and /api/* must be served by the same app on the
    same port. Serving them from separate apps would break the load path."""
    assert client.get("/v1/models").status_code == 200
    assert client.get("/api/v0/models").status_code == 200


# --- quantisation on the load call ------------------------------------------


def test_load_without_quantization_uses_the_registry_default(client):
    """Backward compatible: the field is optional, and omitting it must behave
    exactly as before it existed."""
    resp = client.post("/api/v1/models/load", json={"model": QWEN})

    assert resp.status_code == 200
    assert resp.json()["quantization"] == "int8"  # qwen3's registry default


def test_load_applies_a_quantization_override(client):
    resp = client.post(
        "/api/v1/models/load", json={"model": BGE_BASE, "quantization": "int8"}
    )

    assert resp.status_code == 200
    assert resp.json()["quantization"] == "int8"

    listed = {m["id"]: m for m in client.get("/api/v0/models").json()["data"]}
    assert listed[BGE_BASE]["state"] == "loaded"
    assert listed[BGE_BASE]["quantization"] == "int8"


def test_load_can_turn_quantization_off(client):
    resp = client.post(
        "/api/v1/models/load", json={"model": QWEN, "quantization": "none"}
    )

    assert resp.status_code == 200
    assert resp.json()["quantization"] is None


def test_unrecognised_quantization_still_loads(client):
    """The client falls back to local embedding on any non-2xx, so an unusable
    precision value must not cost the whole batch — it falls back to the
    registry default instead."""
    resp = client.post(
        "/api/v1/models/load", json={"model": QWEN, "quantization": "int3"}
    )

    assert resp.status_code == 200
    assert resp.json()["quantization"] == "int8"


def test_unloaded_models_report_their_registry_quantization(client):
    """`/api/v0/models` describes what a model *would* load as while it isn't
    resident — the client reads this list before deciding to request a load."""
    listed = {m["id"]: m for m in client.get("/api/v0/models").json()["data"]}

    assert listed[QWEN]["state"] == "not-loaded"
    assert listed[QWEN]["quantization"] == "int8"
    assert listed[BGE_BASE]["quantization"] is None
