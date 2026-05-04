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

| Exchange | Type | Routing key | Queue | Consumer |
|---|---|---|---|---|
| `ingestion` | direct | `cpu.chunk.post` | `cpu.chunk.post` | CpuChunkWorker |
| `embedding` | direct | `gpu.embed.{model}` | `gpu.embed.bge-base-v1.5` etc. | GpuEmbedWorker |
| `dlx` | fanout | — | `dlq.*` | (manual review) |

---

## Chunk Tables

Each `(field, model)` pair gets its own table: `chunks_{field}_{model_sanitised}`.
Hyphens in the model name are replaced with underscores.

| task_type | model | table |
|---|---|---|
| `body` | `bge-base-v1.5` | `chunks_body_bge_base_v1_5` |
| `summary_title` | `bge-base-v1.5` | `chunks_summary_title_bge_base_v1_5` |
| `analysis` | `qwen3-0.6b` | `chunks_analysis_qwen3_0_6b` |

---

## Embed Configs (`config/embedding_config.py`)

| task_type | model | vector dim | queue |
|---|---|---|---|
| `body` | `bge-base-v1.5` | 768 | `gpu.embed.bge-base-v1.5` |
| `summary_title` | `bge-base-v1.5` | 768 | `gpu.embed.bge-base-v1.5` |
| `analysis` | `qwen3-0.6b` | 1024 | `gpu.embed.qwen3-0.6b` |
| `query` | `bge-base-v1.5` | 768 | `gpu.embed.bge-base-v1.5` |

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

### Worker Deployment
Workers run as separate processes with their own entrypoints:
- `python -m event_driven_rag_service.worker.entrypoints.cpu`
- `python -m event_driven_rag_service.worker.entrypoints.gpu`

This allows CPU and GPU workers to be scaled independently and keeps resource
profiles separate (CPU workers are IO-bound; GPU workers hold a model in VRAM).

---

## Out of Scope (MVP)

- Semantic search / vector similarity queries
- Inference / categorisation / analysis pipeline
- EmbeddingDispatcher (embeddings written directly by GpuEmbedWorker)
- Auth / multi-tenancy
