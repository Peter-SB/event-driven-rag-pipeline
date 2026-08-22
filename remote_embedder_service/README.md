# remote-embedder

A slim FastAPI service that embeds text on a CUDA GPU and speaks the same HTTP
API LM Studio did. Drop it onto the homelab GPU box, point `EMBED_REMOTE_URL` at
it, and the RAG pipeline cannot tell the difference.

---

## Why this exists

The pipeline's GPU worker offloads embedding to a remote endpoint via
[`remote_embedding.py`](../src/event_driven_rag_service/worker/remote_embedding.py),
falling back to a local model whenever that endpoint is unreachable. That
endpoint used to be LM Studio, which brought problems worth leaving behind:

| LM Studio | remote-embedder |
|---|---|
| Desktop GUI app; loading a model is a click, not a deployment step | Headless process, model set declared in code |
| GGUF builds via llama.cpp | Safetensors via transformers — the canonical weights |
| Undocumented native API that shifts between versions | The API is this repo's, with tests pinning every field |
| No control over batching or residency | Explicit batch size, explicit idle unload |
| Whole desktop app resident to serve one small model | ~200 lines of Python |

**The wire format is unchanged on purpose.** No file under
`src/event_driven_rag_service/` was modified to support this service. The swap
is one environment variable.

---

## The API

Five routes, listed in the order the pipeline's `FallbackEmbeddingModel` calls
them for each batch:

| # | Route | Request | Response |
|---|---|---|---|
| 1 | `GET /v1/models` | — | `{"object": "list", "data": [{"id", "object", "owned_by"}]}` |
| 2 | `GET /api/v0/models` | — | `{"data": [{"id", "state": "loaded"｜"not-loaded", "dim", ...}]}` |
| 3 | `POST /api/v1/models/load` | `{"model": "<id>", "quantization"?: "int8"}` | `{"model", "state": "loaded", "quantization"}` |
| 4 | `POST /v1/embeddings` | `{"model": "<id>", "input": [str, ...]}` | `{"object", "model", "data": [{"index", "embedding"}], "usage"}` |
| 5 | `POST /api/v1/models/unload` | `{"instance_id": "<id>"}` | `{"model", "state": "not-loaded"}` |

Plus `GET /health` for humans and container probes, and `/docs` for the
generated OpenAPI page.

Four constraints fall out of the client code and are worth stating plainly,
because breaking any of them makes the pipeline fall back to local embedding
*silently* rather than erroring:

- **`/v1/*` and `/api/*` must share one origin.** The client derives the native
  API base by stripping the path off `EMBED_REMOTE_URL`.
- **`/v1/embeddings` results are positionally aligned with `input`.** The client
  zips them against its own list by index.
- **Unloaded models are listed by `/api/v0/models` with `state: "not-loaded"`**,
  not omitted. Omitting them reads as "this endpoint doesn't have that model"
  and the load path never fires.
- **`POST /api/v1/models/load` blocks until the model is usable.** The client
  allows `EMBED_REMOTE_LOAD_TIMEOUT_S` (600s default), which covers a first-run
  download.

### By hand

```bash
curl http://localhost:8100/v1/models
curl http://localhost:8100/api/v0/models

curl -X POST http://localhost:8100/api/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-qwen3-embedding-0.6b"}'

curl -X POST http://localhost:8100/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-qwen3-embedding-0.6b","input":["hello","world"]}'

curl -X POST http://localhost:8100/api/v1/models/unload \
  -H 'Content-Type: application/json' \
  -d '{"instance_id":"text-embedding-qwen3-embedding-0.6b"}'
```

---

## Model registry

Defined in [`registry.py`](src/remote_embedder/registry.py). Ids are the same
slugs LM Studio used, so `EMBED_CONFIGS[...].remote_model` in the pipeline needs
no change.

| Wire id | HuggingFace repo | Dim | int8 |
|---|---|---|---|
| `text-embedding-bge-base-en-v1.5` | `BAAI/bge-base-en-v1.5` | 768 | no |
| `text-embedding-bge-small-en-v1.5` | `BAAI/bge-small-en-v1.5` | 384 | no |
| `text-embedding-qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | yes |

Weights download on first load and are cached under `HF_HOME`. Adding a model is
one `ModelSpec` entry; add the matching `remote_model` to the pipeline's
`EMBED_CONFIGS` and `test_registry.py` will hold the two in agreement.

### Requesting a precision

`POST /api/v1/models/load` takes an optional `quantization`, which overrides the
registry default for that load. The pipeline sends its
`EmbedConfig.quantization` there, so precision is declared once — in the
pipeline's config — instead of each side deciding independently and drifting
apart unnoticed.

`"int8"`, `"8bit"`, `"8-bit"`, `"q8"` and `"q8_0"` all mean int8; `"none"`,
`"off"`, `"full"`, `"fp16"` and `"fp32"` all disable it. An unrecognised value
logs a warning and falls back to the registry default rather than failing the
load — the client reads any non-2xx as "embed locally for this batch", so
refusing would trade a precision mismatch for no remote embedding at all.

The response reports what was **actually** applied, which is not always what was
asked for (no CUDA, bitsandbytes missing). Requesting a precision different from
the resident model's forces a genuine reload rather than reusing it.

This is the one place the API deliberately went past LM Studio's. The field is
optional, and a client that omits it gets exactly the old behaviour.

### Why int8 is opt-in per model

int8 is on for Qwen3-Embedding-0.6B and off for the two BGE encoders. That's not
an oversight: bitsandbytes' LLM.int8 does per-layer outlier decomposition, whose
fixed overhead is only amortised by a large enough model. On a 110M-parameter
encoder it typically runs *slower* than fp16 while saving VRAM that was never
scarce. The 0.6B model is where the trade turns favourable.

`quantize` is a hint, not a demand. If bitsandbytes is missing or the quantised
load raises, the service logs a warning and reloads at full precision — a slower
model beats a 503, since the pipeline's own local fallback should be the last
resort rather than the first.

**Safetensors, not GGUF.** The pipeline's *local* fallback leg loads
`Qwen/Qwen3-Embedding-0.6B-GGUF` through llama-cpp-python; this service uses the
plain safetensors repo through transformers. Same model, different packaging.

---

## Batching and idle unload

**Batching.** A request is split into sub-batches of `MAX_BATCH_SIZE` (default
32, matching `GpuEmbedWorker.max_batch`), so one pipeline batch is normally one
forward pass, and an oversized request degrades gracefully instead of OOMing.
Results are reassembled in input order — the tests pin this, because a shuffle
would attach every embedding to the wrong chunk with nothing downstream noticing.

**Residency.** At most one model is resident at a time. A single GPU worker
drives this service and already unloads on model switch, so a second resident
model would cost VRAM for concurrency that never happens. Requesting a different
model evicts the current one.

**Idle unload.** A daemon thread unloads the resident model after
`IDLE_UNLOAD_S` seconds untouched (default 900), then `gc.collect()` and
`torch.cuda.empty_cache()` hand the VRAM back — so the GPU is free for whatever
else the box does between syncs. The next request reloads on demand. Set
`IDLE_UNLOAD_S=0` to disable.

---

## Running it

The GPU box runs this natively — no Docker, no nvidia container runtime, which
is also how LM Studio ran there.

```bash
cd remote_embedder_service

# 1. Install the CUDA build of torch FIRST, matching your driver — the plain
#    `pip install torch` in step 2 would otherwise pull the default CPU wheel.
#    (cu124 shown; check https://pytorch.org for the right index URL, or run
#    `nvidia-smi` and pick the closest cuXXX tag it supports.)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Then the service — bitsandbytes (the int8 path) is a core dependency,
#    installed either way; it just does nothing useful without CUDA.
pip install -e ".[dev]"

# 3. Configure and run.
cp .env.example .env
python -m remote_embedder
```

Serves on `0.0.0.0:8100` by default. Verify with
`curl http://localhost:8100/health`, and check `nvidia-smi` shows the process
after a load.

Single worker on purpose — extra uvicorn workers would each load their own copy
of the model into VRAM.

### Environment

| Variable | Default | Notes |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8100` | |
| `LOG_LEVEL` | `INFO` | |
| `API_KEY` | (empty) | Empty = bearer tokens accepted and ignored. Set to enforce. |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `MAX_BATCH_SIZE` | `32` | Texts per forward pass |
| `IDLE_UNLOAD_S` | `900` | 0 disables auto-unload |
| `REAPER_INTERVAL_S` | `15` | How often the idle check runs |
| `ENABLE_QUANTIZATION` | `1` | 0 forces full precision everywhere |
| `NORMALIZE_EMBEDDINGS` | `1` | L2-normalise output; the pipeline uses cosine distance |
| `HF_HOME` | (HF default) | Weight cache location |
| `LOG_TO_FILE` | `1` | Write logs to `LOG_DIR/LOG_FILE` in addition to the console |
| `LOG_DIR` | `logs` | |
| `LOG_FILE` | `remote_embedder.log` | |
| `LOG_MAX_LINES` | `10000` | Oldest lines dropped once the file exceeds this |

### Logging

This service isn't wired into the pipeline's otel-collector — it's a
standalone process on the GPU box, so the log file is the only persistent
record of what it did. `logs/remote_embedder.log` (by default) is capped at the
last `LOG_MAX_LINES` lines, trimmed in place periodically rather than rotated
into numbered backup files, so disk use stays bounded without external log
rotation.

Only INFO and above is written by default — model load/unload, quantisation
fallback, and request-level warnings/errors. Per-request HTTP access logging is
disabled outright rather than merely downgraded: the pipeline's client fires
`GET /v1/models` before *every* batch, and one log line per health probe would
be routine noise crowding out the events that actually matter. See
[`logging_config.py`](src/remote_embedder/logging_config.py).

---

## Pointing the pipeline at it

In the pipeline's `.env`:

```bash
EMBED_REMOTE_URL=http://<gpu-box>:8100/v1
EMBED_REMOTE_API_KEY=              # only if API_KEY is set here
EMBED_REMOTE_TIMEOUT_S=30.0
EMBED_REMOTE_HEALTH_PATH=/models
EMBED_REMOTE_LOAD_TIMEOUT_S=600.0
```

Note the trailing `/v1` — the client appends `/embeddings` and `/models` to it,
and derives the native `/api/*` base by stripping the path.

Then confirm the GPU worker logs `embeddings generated remotely (endpoint=...)`
rather than `embeddings generated locally (fallback)`.

---

## Tests

```bash
pytest              # contract + manager + registry + real-model smoke; CPU only
pytest -m gpu       # opt-in: real CUDA and the int8 path
```

| File | Covers |
|---|---|
| `test_api_contract.py` | Every field the pipeline's client reads, driven in the order it reads them. Fake encoder — no downloads, no GPU. |
| `test_manager.py` | Batching, residency, eviction, dim verification, idle reaping (injected clock, no sleeps). |
| `test_registry.py` | Cross-checks this registry against the pipeline's `EMBED_CONFIGS`, parsed with `ast` so nothing here imports the pipeline. |
| `test_e2e_smoke.py` | Real bge-small on CPU end to end, including cross-sub-batch ordering. Skips if the weights can't be fetched. |

The registry cross-check earns its keep: the two packages agree only by
convention on a wire id and a vector width, and nothing at runtime notices when
they drift — a renamed id just makes the GPU worker embed locally forever, and a
changed dimension corrupts a pgvector column.
