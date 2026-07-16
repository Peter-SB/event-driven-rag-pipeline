# CONTEXT.md — Domain Glossary & Design Decisions

This document records the resolved terminology and key design choices for the
event-driven-rag-pipeline. It is the single source of truth for naming
conventions used across code, events, tasks, and infrastructure.

---

## Core Entities

| Term | Definition |
|---|---|
| **Post** | Source-agnostic content unit (Reddit post, article, note, …). Identified by `external_id` + `external_source`. The primary key is `post_id` (auto-incremented in Postgres). |
| **Chunk** | A text window derived from a Post field, targeted at ~500 words with ~10% overlap. Identified by a UUID `id`. Carries `text_hash` for deduplication. |
| **Embedding** | A dense float vector stored in the same row as the Chunk (nullable until the GPU worker fills it). Vector dimension is model-dependent (see Embed Config). |

---

## Pipeline Stages

```
[Client]  POST /posts/sync
    │
    ▼
[API]  PostRepository.upsert()
    │  publish post.synced → event log
    ▼
[PostDispatcher]  subscribe post.synced
    │  enqueue ChunkTask → cpu.chunk.post (RabbitMQ)
    ▼
[CpuChunkWorker]  chunk_at_boundaries → ChunkRepository.bulk_insert()
    │  publish chunks.created → event log
    ▼
[ChunkDispatcher]  subscribe chunks.created
    │  enqueue EmbedTask → gpu.embed.{model} (RabbitMQ)
    ▼
[GpuEmbedWorker]  SentenceTransformer.encode() → ChunkRepository.update_embeddings()
    │  publish embedding.completed → event log
    ▼
[future: SearchDispatcher / AnalysisDispatcher]
```

---

## Event Log Topics (Redpanda / Postgres event_log)

| Topic | Published by | Consumed by |
|---|---|---|
| `post.synced` | API sync route | PostDispatcher |
| `chunks.created` | CpuChunkWorker | ChunkDispatcher |
| `embedding.completed` | GpuEmbedWorker | (future) SearchDispatcher |

---

## Task Queue Exchanges & Routing (RabbitMQ)

> Source of truth: `infrastructure/task_queue.py` (BINDINGS) / `config/embedding_config.py`
> (EMBED_CONFIGS) — verify against those before trusting this table, it's a hand-maintained
> mirror with no automated check tying it to the code.

Note `embedding` is a **topic** exchange, not direct: multiple task_types can route to the
same queue under different model strings (see `summary_title`/`analysis` below sharing
`gpu.embed.qwen3-0.6b`), which only topic routing supports. The routing key published is
always `EMBED_CONFIGS[task_type].queue` — never derived from the model name (see
`tasks/registry.py` docstring for why that footgun was removed).

| Exchange | Type | Routing key | Queue | Consumer |
|---|---|---|---|---|
| `ingestion` | topic | `cpu.chunk.post` | `cpu.chunk.post` | CpuChunkWorker |
| `embedding` | topic | `EMBED_CONFIGS[task_type].queue` | `gpu.embed.bge-base-en-v1.5`, `gpu.embed.bge-small-en-v1.5`, `gpu.embed.qwen3-0.6b` | GpuEmbedWorker |
| `dlx` | direct | `dlq.{queue}` | `dlq.*` | (manual review) |

---

## Chunk Tables

Each `(post_table, task_type, model)` triple gets its own table, built by
`utils/build_table_names.build_chunk_table_name()`:
`{post_table}_chunks_{task_type}_{model_sanitised}`. Never hardcode this — always call the
builder, and never hand-maintain a literal example elsewhere (Grafana dashboards learned
this the hard way once already).

| task_type | model | example table (`post_table="posts_main"`) |
|---|---|---|
| `body` | `BAAI/bge-base-en-v1.5` | `posts_main_chunks_body_baai_bge_base_en_v1_5` |
| `title` | `BAAI/bge-small-en-v1.5` | `posts_main_chunks_title_baai_bge_small_en_v1_5` |
| `summary_title` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `posts_main_chunks_summary_title_qwen3_embedding_0_6b_q8_0_gguf` |
| `analysis` | `Qwen/Qwen3-0.6B` | `posts_main_chunks_analysis_qwen_qwen3_0_6b` |

---

## Embed Configs (`config/embedding_config.py`)

`summary_title` and `analysis` intentionally share one GPU queue
(`gpu.embed.qwen3-0.6b`) under two different model strings — the queue is a physical
worker/model-load unit, not a 1:1 mirror of `model`.

| task_type | model | vector dim | queue |
|---|---|---|---|
| `body` | `BAAI/bge-base-en-v1.5` | 768 | `gpu.embed.bge-base-en-v1.5` |
| `title` | `BAAI/bge-small-en-v1.5` | 384 | `gpu.embed.bge-small-en-v1.5` |
| `summary_title` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 1024 | `gpu.embed.qwen3-0.6b` |
| `analysis` | `Qwen/Qwen3-0.6B` | 1024 | `gpu.embed.qwen3-0.6b` |

---

## Consumer Groups (`config/consumer_groups.py`)

| Constant | Value | Used by |
|---|---|---|
| `POST_SYNCED` | `post-dispatcher-synced` | PostDispatcher |
| `POST_ANALYSED` | `post-dispatcher-analysed` | (future) |
| `CHUNKS_CREATED` | `chunk-dispatcher-chunks-created` | ChunkDispatcher |
| `EMBEDDING_COMPLETED` | `embedding-dispatcher-completed` | (future) |

---

## Key Design Decisions

### Idempotency
- **Upsert freshness**: `post_repository.upsert()` only updates a row if the incoming
  `updated_at > existing updated_at`. Returns `"skipped"` otherwise.
- **Chunk deduplication**: `CpuChunkWorker` checks `text_hash` against existing rows
  before inserting. Unchanged chunks are not re-inserted, so embeddings are preserved.

### fields_changed (ECST pattern)
`PostSyncedEvent.fields_changed: list[str]`
- **Empty list** (`[]`) → post is new; all fields should be chunked.
- **Non-empty list** → only the listed fields changed; dispatcher can skip other chunk tasks.
- PostDispatcher currently dispatches `body` and `summary_title` tasks regardless
  (CpuChunkWorker's `text_hash` check handles the actual dedup).

### Event Bus
Two implementations behind `EventBusBase`:
- **PostgresEventBus** (homelab): polls `event_log` table at 500 ms intervals. Simple
  to operate, no extra infrastructure.
- **RedpandaEventBus** (production): persistent Kafka producer, consumer groups via
  `group_id`.
  
Switch via `EVENT_BUS=redpanda` environment variable.

### Post Model
Source-agnostic: `external_id`, `external_source`, `external_created_at`.
`subreddit` is nullable. The client-facing wire format uses camelCase aliases
(`redditId`, `externalSource`, …) so the Reddit sync client requires no changes.

### Post updated_at
The `updated_at` field is used for freshness and comes from the source, never use the system time.

### Worker Deployment
Workers run as separate processes with their own entrypoints:
- `python -m event_driven_rag_service.worker.entrypoints.cpu`
- `python -m event_driven_rag_service.worker.entrypoints.gpu`
- `python -m event_driven_rag_service.worker.entrypoints.dispatcher`

This allows CPU and GPU workers (and dispatchers) to be scaled independently and keeps
resource profiles separate (CPU workers are IO-bound; GPU workers hold a model in VRAM;
dispatchers are event-loop-bound).

Dispatchers bridge the event log (Redpanda/Postgres) to RabbitMQ. The combined dispatcher
entrypoint runs PostDispatcher and ChunkDispatcher concurrently in a single process.

### Analysis Pipeline (Deferred)
The `analysis_text` field in ChunkTask is currently unused and will be removed before
the analysis pipeline is implemented. For MVP, only `body` and `summary_title` chunking
are active.

---

## Out of Scope (MVP)

- Semantic search / vector similarity queries
- Inference / categorisation / analysis pipeline (analysis chunk tasks deferred)
- SearchDispatcher (triggered post embedding.completed events — see future note above)
- Auth / multi-tenancy
