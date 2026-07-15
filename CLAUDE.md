# CLAUDE.md — Project Context for AI Assistants

This document provides architecture context, key design decisions, and gotchas to help AI assistants (including Claude) navigate, understand, and contribute to the event-driven RAG pipeline project.

---

## Project Overview

**Event-Driven Cloud-Sync and RAG Pipeline** is a two-layer distributed system for syncing, chunking, embedding, and searching documents. It's designed to be both production-ready and runnable on a homelab with Docker Compose.

**Key goals:**
- Clean separation between Event Log (immutable, replayable) and Task Queue (ephemeral, routed work)
- Production-inspired architecture that scales from homelab to Kubernetes
- Observable, testable, extensible codebase

---

## Architecture at a Glance

### Two Layers

```
Event Log (Kafka/Redpanda/Postgres mock)
    ↓ [consumers]
Dispatchers (translate events → tasks)
    ↓ [publish to]
Task Queue (RabbitMQ)
    ↓ [consumed by]
Workers (CPU: chunking; GPU: embedding)
    ↓ [persist to]
Postgres + pgvector
```

### MVP Scope (Current)

**Active:** Post sync → chunking → embedding  
**Deferred:** Search, analysis, inference

### Key Components

| Component | Files | Responsibility |
|-----------|-------|-----------------|
| **API** | `api/app.py`, `api/sync.py` | FastAPI server, POST /posts/sync endpoint |
| **Repositories** | `repository/{post,chunk}_repository.py` | Async Postgres access (asyncpg) |
| **Handlers** | `handlers/{chunk,embed}_handler.py` | Business logic: receive task → do work → emit event |
| **Workers** | `worker/{cpu,gpu}_worker.py` | Long-lived processes, consume RabbitMQ, delegate to handlers |
| **Dispatchers** | `dispatchers/{post,chunk,embedding,search}_dispatcher.py` | Bridge event log → RabbitMQ (translate events to tasks) |
| **Event Bus** | `infrastructure/event_bus.py` | Abstract event log (Redpanda vs Postgres) |
| **Tasks** | `tasks/{chunk,embed}_task.py` | Pydantic schemas for RabbitMQ messages |
| **Events** | `events/{post,chunk}_events.py` | Pydantic schemas for event log topics |

---

## Critical Design Decisions

### 1. Two-Layer Architecture (Event Log + Task Queue)

**Why:** Solves two different problems:
- **Event Log** is the source of truth (durable, replayable, audit trail)
- **Task Queue** is work distribution (routing, retries, backpressure)

Neither is redundant. Never collapse them into one.

**Impact:** Dispatchers are the only component aware of both layers. This is intentional and desirable — it keeps producers and consumers decoupled.

### 2. Async Asyncpg (Not Sync Psycopg2)

**Why:** Workers run inside an asyncio event loop. Synchronous psycopg2 blocks the loop, defeating the purpose of async workers.

**Gotcha:** `post_repository.py` had psycopg2 code mixed in (now removed). If you see `import psycopg2`, it's a bug.

**Rule:** All DB access is async via asyncpg. Never use sync DB calls inside async workers.

### 3. Library-Scoped Table Names (Per-Client Isolation)

**Why:** Each client library gets its own post table. This ensures complete data isolation and allows the system to serve multiple libraries simultaneously.

**Pattern:** 
- Post table: `posts_{library_id}` (e.g., `posts_main`, `posts_work`)
- `library_id` always comes from the client (via POST /posts/sync)

**Impact:** 
- `PostFetcher.fetch()` and `PostRepository` methods accept `table_name` as a parameter
- `post_table` flows on every event and task throughout the pipeline
- No server-side default table names

### 4. Chunk Table Naming: `posts_{id}_chunks_{field}_{model_sanitised}`

**Pattern:** `posts_main_chunks_body_bge_base_v1_5` (hyphens → underscores)

**Why:** Each `(library, field, model)` triple gets its own table because:
- Data is scoped per library (complete isolation)
- Different embedding models have different vector dimensions
- Separate tables avoid mixing different embedding approaches
- Scales naturally (add new library/field/model = add new table)

**Gotcha:** The table name is derived from `post_table`, `task_type`, and `embed_model`. Never hardcode table names — always use `build_chunk_table_name()`.

### 5. Dispatcher Entrypoint Pattern

**Design:** Single combined entrypoint runs PostDispatcher + ChunkDispatcher concurrently.

```python
# Single process, two async tasks
dispatcher = PostDispatcher(rmq, event_bus)
chunk_dispatcher = ChunkDispatcher(rmq, event_bus)
await asyncio.gather(dispatcher.run(), chunk_dispatcher.run())
```

**Why:** For homelab simplicity. In production, you'd split them into separate containers.

**Location:** `worker/entrypoints/dispatcher.py`

### 6. Lazy Chunk Table Creation

**Pattern:** Chunk tables are created on-demand when the first chunk task arrives, not at startup.

```python
# In ChunkPostHandler.handle():
vector_dim = EMBED_CONFIGS[task.task_type].dim
await self._chunks.ensure_table(chunk_table, vector_dim)
```

**Why:** Chunk tables are per-library and can't be pre-created at startup. This approach allows unlimited libraries without pre-configuration.

**Location:** `handlers/chunk_handler.py` (in `handle()` method)

### 7. BaseEvent Config: `extra="forbid"`

**Setting:** `model_config = ConfigDict(extra="forbid")`

**Why:** Strict validation. Rejects unknown fields, catches bugs early.

**Impact:** Event payloads must match the schema exactly. No extra fields allowed.

---

## Common Gotchas

### Gotcha 1: Post Table Must Come From Client

**The mistake:** Assuming the server can use a default post table or that the same table is always used.

**Reality:** The client MUST provide `library_id` in every POST /posts/sync request. The server builds `post_table = f"posts_{library_id}"` and passes it through all repository methods.

**Fix:** All `PostRepository` methods accept `table_name` parameter. Never rely on defaults.

### Gotcha 2: Chunk Table Creation Happens on First Task

**The mistake:** Assuming chunk tables exist before the first chunk task arrives.

**Reality:** Chunk tables are created lazily by `ChunkPostHandler` when it receives the first task for that library+field+model combination. The table name is derived: `posts_{id}_chunks_{type}_{model}`.

**Debug:** Check handler logs — if you see `"ChunkRepository: table 'posts_main_chunks_body_...' ready"`, the table was just created.

### Gotcha 3: Event Bus Swapping

**The mistake:** Assuming the event bus implementation is always Redpanda.

**Reality:** It's selected via `EVENT_BUS` env var:
- `EVENT_BUS=postgres` (default, homelab, uses Postgres mock)
- `EVENT_BUS=redpanda` (production, uses Kafka-compatible Redpanda)

**Impact:** All event bus interaction is behind `EventBusBase`. The implementation is pluggable. Unit tests use Postgres mock even in production builds.

### Gotcha 4: Text Hash Deduplication

**The mistake:** Assuming `POST /sync` with identical text = zero chunks stored.

**Reality:** Chunks are deduplicated by `text_hash` within a post. If the same text is synced twice, `CpuChunkWorker` detects this and skips re-insertion.

**Impact:** Embeddings are preserved across re-syncs (idempotent).

### Gotcha 5: analysis_text in ChunkTask

**The mistake:** Wondering why `analysis_text` exists but is never populated.

**Reality:** It's deferred for post-MVP. Analysis pipeline hasn't been built yet. The field exists to avoid schema migration but should be ignored.

**Note:** See `CONTEXT.md` for deferral note.

---

## Key Files & Patterns

### Repository Pattern (Async Protocols)

**Files:** `repository/post_repository.py`, `repository/chunk_repository.py`

**Pattern:** Both implement async protocols (`fetch`, `bulk_insert`, `ensure_table`). Tests inject fake implementations.

**Rule:** Never call `.sync_to_async()` or `.run_until_complete()` on repository methods inside async context. The repository is already async.

### Task / Event Schemas

**Location:** `tasks/*.py`, `events/*.py`

**Pattern:** Pydantic models with `ConfigDict(populate_by_name=True)` for backward compatibility.

**Rule:** Always use the `to_dict()` method when publishing to RabbitMQ or event log. Never use `model_dump()` directly (different serialization).

### Handler Pattern

**Location:** `handlers/*.py`

**Pattern:** Stateless handler class with a single async `handle()` method. Repositories injected at construction.

**Example:**
```python
class ChunkPostHandler:
    def __init__(self, post_fetcher, chunk_store, version_checker, event_log):
        self._posts = post_fetcher
        self._chunks = chunk_store
        self._versions = version_checker
        self._event_log = event_log
    
    async def handle(self, task: ChunkTask) -> list[str]:
        # Fetch post, chunk text, check hashes, insert, emit event
```

### Worker Pattern

**Location:** `worker/{cpu,gpu}_worker.py`

**Pattern:** Long-lived synchronous worker (pika consumer). Bridges sync pika to async handler via `loop.run_until_complete()`.

```python
def _on_message_callback(self, ch, method, properties, body):
    task = ChunkTask.model_validate_json(body)
    chunk_ids = self._loop.run_until_complete(
        self._handler.handle(task)
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)
```

---

## Testing Approach

### Three Layers

1. **Unit:** Handlers with mocked repositories (fast, isolated)
2. **Integration:** Real Postgres, mocked RabbitMQ (DB-aware)
3. **E2E:** Full docker-compose stack (end-to-end pipeline)

### Mocking Pattern

**Fake repositories:** In-memory implementations of protocols (used in unit tests)  
**Mock models:** Deterministic mock SentenceTransformer (used everywhere, no GPU needed)

### Key Test Files

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures (db_pool, test_event_bus, mock_embedding_model) |
| `tests/unit/test_chunk_handler.py` | ChunkPostHandler with fake PostFetcher |
| `tests/integration/test_chunk_repository.py` | ChunkRepository against real Postgres |
| `tests/e2e/test_sync_api.py` | Full pipeline: POST /sync → chunks → embeddings |

---

## Environment Variables

| Variable | Default | Used by | Notes |
|----------|---------|---------|-------|
| `DB_URL` | — | All | Postgres connection string (required) |
| `RABBITMQ_URL` | — | Workers, dispatchers | RabbitMQ connection (required) |
| `REDPANDA_SERVERS` | — | EventBus | Comma-separated broker list (optional, for Redpanda) |
| `EVENT_BUS` | `postgres` | All | `postgres` or `redpanda` |
| `MOCK_EMBEDDINGS` | (unset) | GPU worker | If set, use deterministic mock model (for dev/CI) |
| `EMBED_REMOTE_URL` | (unset) | GPU worker | OpenAI-compatible base URL (e.g. LM Studio) to try before the local model; unset disables remote embedding |
| `EMBED_REMOTE_API_KEY` | (unset) | GPU worker | Optional bearer token sent to the remote endpoint |
| `EMBED_REMOTE_TIMEOUT_S` | `10.0` | GPU worker | Per-request timeout for the remote endpoint |
| `EMBED_REMOTE_HEALTH_INTERVAL_S` | `30.0` | GPU worker | Background health-check poll interval; a failed live request also demotes status immediately |

---

## Common Tasks

### Adding a New Chunk Type (e.g., `metadata`)

1. Add to `EMBED_CONFIGS` in `config/embedding_config.py`
2. Update `PostDispatcher._dispatch_chunk_tasks()` to check for the new type
3. Update `ChunkPostHandler._resolve_text()` to handle the new type
4. No need to pre-create tables — they're created lazily on first task arrival

### Syncing Posts With a New Library

Every POST /posts/sync request must include a `library_id`:

```json
{
  "library_id": "main",
  "posts": [...]
}
```

This creates the post table `posts_main` on first sync. Subsequent syncs with the same `library_id` reuse the existing table. Different `library_id` values create entirely separate post and chunk tables.

### Adding a New Event Type

1. Define schema in `events/*.py` (extend `BaseEvent`)
2. Publisher emits via `event_bus.publish(event_type, event.to_dict())`
3. Consumer subscribes via `event_bus.subscribe(event_type, consumer_group=...)`
4. Update CONTEXT.md with new topic

### Switching to Redpanda (from Postgres mock)

1. `docker-compose up redpanda` (or use managed Redpanda)
2. `export EVENT_BUS=redpanda REDPANDA_SERVERS=localhost:29092`
3. All code works unchanged (pluggable EventBusBase)

---

## Performance Notes

### GPU Worker Batching

GPU worker batches embeddings by default (configurable in `EmbedHandler`). Larger batches = higher throughput but higher latency. Sweet spot is usually 32–128 texts per batch.

### Chunk Deduplication

`text_hash` check prevents re-embedding unchanged chunks. This is critical for idempotency and cost savings when re-syncing posts.

### Consumer Group Offset

If a dispatcher crashes, it rejoins its consumer group and resumes from the last committed offset. Unprocessed events are replayed (exactly-once semantics via consumer group).

---

## References

- **Architecture:** See [docs/Architecture.md](./docs/Architecture.md)
- **Terminology:** See [CONTEXT.md](./CONTEXT.md)
- **Testing:** See [TESTS.md](./TESTS.md)
- **Glossary:** See [docs/Glossary.md](./docs/Glossary.md)
