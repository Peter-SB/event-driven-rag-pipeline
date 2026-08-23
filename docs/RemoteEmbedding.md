# Remote Embedding

How the GPU worker offloads embedding to a separate machine, what happens when
that machine is off, and why the arrangement is shaped this way.

---

## The problem

Embedding is the only genuinely GPU-hungry step in the pipeline. The homelab
machines that run the rest of the stack — API, dispatchers, CPU chunking worker,
Postgres, RabbitMQ — have no GPU worth the name, while a gaming PC on the same
Tailscale network does.

Two obvious approaches, both wrong:

- **Run the whole stack on the GPU box.** Ties the pipeline's availability to a
  machine that gets rebooted into games. Sync should keep working when it's off.
- **Make the GPU box a hard dependency of the GPU worker.** Same problem, and
  every sync would block on a machine that isn't guaranteed to be awake.

What's wanted instead: use the GPU when it's there, degrade to slow local
embedding when it isn't, and never make the caller think about which happened.

---

## The shape

```
GpuEmbedWorker
    ↓ .encode(texts)
FallbackEmbeddingModel          ← implements the EmbeddingModel protocol
    ├── RemoteEmbeddingModel    → HTTP → remote-embedder service (CUDA box)
    └── local model             → SentenceTransformer, in-process
```

`FallbackEmbeddingModel` is a drop-in `EmbeddingModel`: it exposes `.name` and
`.encode()`, and nothing above it — not `GpuEmbedWorker`, not `EmbedHandler` —
knows a remote endpoint exists. That's the whole trick. Remote embedding is a
substitution at one construction site
([`worker/entrypoints/gpu.py`](../src/event_driven_rag_service/worker/entrypoints/gpu.py)),
not a concern threaded through the pipeline.

Source:
[`worker/remote_embedding.py`](../src/event_driven_rag_service/worker/remote_embedding.py).

---

## The pre-flight ladder

Before every batch, three checks run in order. Each can send the batch to the
local model without failing the request:

1. **Is the endpoint reachable?** `GET {EMBED_REMOTE_HEALTH_PATH}` against
   `EMBED_REMOTE_URL`. Failure means the box is off or the network is down —
   go local.
2. **Does the endpoint serve this model at all?** `GET /api/v0/models`, looking
   for the id. Failure here means the endpoint is healthy but doesn't have this
   particular model, which is a different situation and must not mark the
   endpoint as down.
3. **Is the model resident?** The `state` field from the same listing. If it
   reads `not-loaded`, request `POST /api/v1/models/load` and wait.

Only when all three pass does `POST /embeddings` get the batch. A failure there
also falls back to local.

### Cold starts and the load timeout

Step 3 is the slow rung, and its cost is bimodal. A model already downloaded on
the remote loads in seconds; a model the remote has never seen must first be
pulled from HuggingFace — around 1.2GB for Qwen3-Embedding-0.6B — before
anything touches the GPU. Nothing needs pre-seeding for this: the service
downloads on demand, and lists models it has never downloaded as `not-loaded`
rather than omitting them, precisely so step 2 passes and step 3 gets the chance
to fetch them.

`EMBED_REMOTE_LOAD_TIMEOUT_S` therefore defaults to **600s**, against
`EMBED_REMOTE_TIMEOUT_S`'s 10s for inference. Overshooting costs nothing — the
call returns the moment the model is ready — whereas undershooting makes every
cold start look like a failure.

Timing out is not fatal, and is logged as such. The server keeps loading after
the client gives up, so the next batch's pre-flight usually finds the model
resident; the warning says so explicitly rather than reading as a hard error.
The batch that timed out goes to the local model, as any other rung failure
would.

> A ten-minute blocking call with no output is indistinguishable from a hang, so
> the load is logged when it *starts* — with the model, the requested precision
> and the timeout budget — not only when it finishes.

### Why check before every batch rather than polling

A batch is already the natural unit of work — a background poller would add a
thread, a staleness window, and a race between "last poll said healthy" and
"the box went to sleep just now", in exchange for saving one cheap GET per batch
of up to 32 embeddings. Not a trade worth making.

### Why `/api/v0/models` and not `/v1/models`

The OpenAI-compatible `GET /v1/models` lists every model the server has
*downloaded*, whether or not it is loaded into VRAM. It is a reachability
signal, not a usability one. `/api/v0/models` reports a real `state` field, so
step 3 can tell "needs loading" from "ready".

This distinction was inherited from LM Studio and deliberately kept: it is
genuinely the right split, and keeping it meant the pipeline needed no changes
when LM Studio was replaced.

---

## Which models go remote

Each entry in [`config/embedding_config.py`](../src/event_driven_rag_service/config/embedding_config.py)
declares independently whether either leg is permitted.

| Chunk type | `primary` | `remote_model` | `quantization` | Legs |
|---|---|---|---|---|
| `body` | `BAAI/bge-base-en-v1.5` | `text-embedding-bge-base-en-v1.5` | — | remote + local |
| `title` | `BAAI/bge-small-en-v1.5` | `text-embedding-bge-small-en-v1.5` | — | remote + local |
| `analysis` | `Qwen/Qwen3-Embedding-0.6B` | `text-embedding-qwen3-embedding-0.6b` | `int8` | remote + local |
| `summary_title` | `Qwen/Qwen3-Embedding-4B` | `text-embedding-qwen3-embedding-4b` | `int8` | remote only |

`allow_remote` and `allow_local` are separate booleans rather than one mode,
because the four combinations are genuinely different situations:

| `allow_remote` | `allow_local` | Behaviour |
|---|---|---|
| yes | yes | The default. Remote first, local when the endpoint is down. |
| yes | no | Remote only. A failed batch is **skipped** for the backfill endpoint rather than embedded locally. |
| no | yes | Never leaves this box. |
| no | no | Config error — raised at worker startup, not at first task. |

The remote-only case is the one worth explaining. Falling back to a local model
sounds strictly safer, but for a model the box can't really run it isn't: the
worker would spend minutes per batch on CPU, and the vectors would still land in
the same pgvector column as the remote ones.

So the batch is *skipped* instead. `_NoLocalFallback.encode()` raises
`EmbeddingUnavailableError`, and `EmbedHandler` treats that specifically: it
warns, **acks** the tasks, and persists nothing. The chunk rows are already in
the table with `embedding IS NULL`, so the work is not lost — it is exactly what
`POST /maintenance/requeue-missing-embeddings` scans for. Bring the endpoint
back, call the endpoint, and the outage window drains.

Acking rather than nacking is the deliberate part. Nacking would dead-letter
every message from the outage, turning a temporary endpoint being down into a
pile of messages someone has to replay by hand. Only *queries* still fail
outright on this path — a query has no persisted row to backfill, and a search
waiting on an embedding needs to hear about it now.

> A model is only sent remotely when both sides have been told about it —
> `remote_model` here, and the registry in `remote_embedder_service`.

`remote_model` is separate from `primary` because the remote id need not match
the canonical one; the service keys its registry on the slugs LM Studio used, so
those ids survived the swap. `.name` on the wrapper always reports `primary`,
whichever leg served the batch, because that is what chunk table names, queue
routing and query-prefix lookups are all keyed on.

### Quantisation

`quantization` is declared once, in the pipeline's config, and both legs honour
it the same way: bitsandbytes at load time. Remotely it travels on the load call
as `{"model": ..., "quantization": "int8"}`; locally it becomes a
`BitsAndBytesConfig(load_in_8bit=True)` in `SentenceTransformer`'s
`model_kwargs`. Same checkpoint, same precision, either way — a batch that falls
back mid-run is comparable with the ones around it in the pgvector column.

It is advisory on both legs. A box that can't honour it — no CUDA device,
bitsandbytes missing — loads at full precision and says so (the remote in its
load response, the local in its log line) rather than failing. Vectors at the
wrong precision are still usable; no vectors at all are not.

The accepted spellings (`int8`, `8bit`, `q8_0`, …) are duplicated in
`worker/entrypoints/gpu.py` and `remote_embedder_service/manager.py`. They must
agree: one config string that means int8 on one leg and full precision on the
other is worse than no quantisation at all.

> This is the one place the wire format deliberately went past what LM Studio
> did. The field is optional, and a server that ignores it behaves exactly as
> before.

---

## The endpoint: remote-embedder

The remote is [`remote_embedder_service/`](../remote_embedder_service/) — a
small FastAPI service running CUDA on the gaming PC. See its
[README](../remote_embedder_service/README.md) for the API, the model registry,
and how to run it.

### Why not LM Studio

LM Studio filled this role first, and it was a reasonable way to find out
whether remote embedding was worth doing at all. It was a poor fit to keep:

- A desktop GUI application on a headless-ish dependency. Loading a model was a
  click, not a deployment step, so the service's state lived in someone's memory
  rather than in code.
- GGUF via llama.cpp, when the pipeline wants the canonical safetensors weights
  and the GPU can run them directly. (The pipeline's local leg briefly loaded
  GGUF too, via llama-cpp-python; it now loads the same safetensors repo as the
  remote, so the two legs no longer disagree about what "the model" is.)
- An undocumented native API (`/api/v0/models`, `/api/v1/models/load`) that
  shifts between versions, with nothing pinning the fields the pipeline depends
  on.
- No control over batch size, model residency, or when VRAM gets released.

The replacement keeps the wire format byte-for-byte and changes everything
behind it. No file under `src/event_driven_rag_service/` was modified — the
migration is one environment variable, which is the strongest evidence the
abstraction boundary was in the right place.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `EMBED_REMOTE_URL` | (unset) | Base URL including `/v1`, e.g. `http://gpu-box:8100/v1`. Unset disables remote embedding entirely. |
| `EMBED_REMOTE_API_KEY` | (unset) | Sent as `Authorization: Bearer`. Safe to leave unset. |
| `EMBED_REMOTE_TIMEOUT_S` | `10.0` | Per-request timeout for health and embed calls. |
| `EMBED_REMOTE_HEALTH_PATH` | `/models` | Relative to `EMBED_REMOTE_URL`. |
| `EMBED_REMOTE_LOAD_TIMEOUT_S` | `600.0` | Model-load calls only, and far slower than inference — a first load downloads weights from HuggingFace before touching the GPU. |

Read in [`config/settings.py`](../src/event_driven_rag_service/config/settings.py).

The trailing `/v1` matters. The client appends `/embeddings` and `/models` to
the URL as given, and derives the native `/api/*` base by stripping the path —
so both surfaces must be served from the same origin and port.

---

## Observability

There is no metric for "was this batch embedded remotely". The signal is in the
GPU worker's logs, one line per batch:

```
embeddings generated remotely (endpoint=gpu-box:8100 count=32 model=BAAI/bge-base-en-v1.5)
embeddings generated locally (fallback) (count=32 model=BAAI/bge-base-en-v1.5)
```

Worth knowing when debugging slow syncs: **silent fallback is the expected
failure mode.** Every remote problem — box asleep, model missing, endpoint
returning garbage — surfaces as embedding merely being slower, never as an
error. If a sync is dragging, check which of those two lines is being written
before looking anywhere else.

---

## Testing

| Layer | File | What it proves |
|---|---|---|
| Unit | [`tests/unit/test_remote_embedding.py`](../tests/unit/test_remote_embedding.py) | The pre-flight ladder, via `httpx.MockTransport`. Every rung's failure falls back to local. |
| Unit | [`tests/unit/test_gpu_entrypoint_remote.py`](../tests/unit/test_gpu_entrypoint_remote.py) | The routing decision, parametrised over every `EMBED_CONFIGS` entry. |
| Integration | [`tests/integration/test_remote_embedding_pipeline.py`](../tests/integration/test_remote_embedding_pipeline.py) | Remotely produced vectors reach Postgres intact. |
| Live | [`tests/integration/test_remote_embedding_live.py`](../tests/integration/test_remote_embedding_live.py) | The real endpoint. Marked `remote_embedding`, skipped unless `EMBED_REMOTE_URL` is set. |
| Service | [`remote_embedder_service/tests/`](../remote_embedder_service/tests/) | The endpoint's own contract, batching, residency and a real-model smoke test. |

```bash
pytest tests/unit/test_remote_embedding.py            # no endpoint needed
EMBED_REMOTE_URL=http://gpu-box:8100/v1 pytest -m remote_embedding
cd remote_embedder_service && pytest                  # the service's own suite
```

The two suites are held together by
`remote_embedder_service/tests/test_registry.py`, which parses this repo's
`EMBED_CONFIGS` and asserts every `remote_model` id is servable at the declared
dimension, and that every `quantization` string is one the service recognises.
The packages are installed separately and agree only by convention on those few
strings per model; nothing at runtime notices when they drift.

---

## Migration: the Qwen3 chunk tables were renamed

`EmbedConfig.primary` replaced the old `model` field, and for Qwen3 its value
changed from the GGUF filename to the canonical repo id. (GGUF has since gone
entirely: both legs now load the safetensors repo.) Chunk table names embed
that identity, so they moved:

| Old | New |
|---|---|
| `posts_{lib}_chunks_summary_title_qwen3_embedding_0_6b_q8_0_gguf` | `posts_{lib}_chunks_summary_title_qwen_qwen3_embedding_0_6b` |
| `posts_{lib}_chunks_analysis_qwen3_embedding_0_6b_q8_0_gguf` | `posts_{lib}_chunks_analysis_qwen_qwen3_embedding_0_6b` |

The `body` and `title` tables are unaffected — those configs already used their
HF repo id.

Nothing breaks if you skip this: chunk tables are created lazily, so the
pipeline will simply build the new ones empty and re-embed on the next sync,
leaving the old tables orphaned. Renaming is only worth doing to keep existing
embeddings, which avoids re-running the Qwen3 model over every summary:

```sql
-- Per library. Run before starting the workers on the new config.
ALTER TABLE posts_main_chunks_summary_title_qwen3_embedding_0_6b_q8_0_gguf
  RENAME TO posts_main_chunks_summary_title_qwen_qwen3_embedding_0_6b;
ALTER TABLE posts_main_chunks_analysis_qwen3_embedding_0_6b_q8_0_gguf
  RENAME TO posts_main_chunks_analysis_qwen_qwen3_embedding_0_6b;
```

To find every table that needs it across all libraries:

```sql
SELECT tablename FROM pg_tables
WHERE tablename LIKE '%\_chunks\_%qwen3\_embedding\_0\_6b\_q8\_0\_gguf';
```

The vectors themselves are unchanged — same model, same 1024 dimensions, same
weights. Only the name the pipeline looks them up under moved.

---

## See also

- [remote_embedder_service/README.md](../remote_embedder_service/README.md) — the service itself
- [Architecture.md](./Architecture.md) — where the GPU worker sits
- [Testing.md](./Testing.md) — the three-layer testing approach
