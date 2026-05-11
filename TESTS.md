# TESTS.md — Testing Strategy & Fixtures

This document covers the testing approach, fixture setup, and running tests for the event-driven RAG pipeline.

---

## Testing Layers

The project uses a three-layer testing strategy:

### 1. Unit Tests
**Scope:** Isolated business logic with mocked dependencies  
**Purpose:** Verify core logic correctness (handlers, repositories, dispatchers)  
**Examples:**
- `test_chunk_handler.py` — ChunkPostHandler with fake PostFetcher, ChunkStore
- `test_cpu_worker.py` — CpuChunkWorker with mocked RabbitMQ
- `test_embed_handler.py` — EmbedHandler with fake ChunkFetcher, EmbeddingStore

### 2. Integration Tests
**Scope:** Real Postgres, real RabbitMQ test containers  
**Purpose:** Verify component interaction and data persistence  
**Examples:**
- `test_chunk_repository.py` — ChunkRepository against real Postgres schema
- `test_gpu_worker.py` — GPU worker startup, table auto-creation, embedding persistence

### 3. End-to-End Tests
**Scope:** Full docker-compose stack, all infrastructure  
**Purpose:** Verify complete pipeline flow (POST /sync → chunks → embeddings)  
**Examples:**
- `test_sync_api.py` — POST /sync with live API, event bus, workers
- `test_ingest_pipeline.py` — Full ingest pipeline with polling verification

---

## Test Directory Structure

```
tests/
├── conftest.py                          # Shared fixtures (DB pools, event bus, mocked models)
├── unit/
│   ├── test_cpu_worker.py               # CpuChunkWorker + ChunkPostHandler
│   ├── test_chunk_handler.py            # ChunkPostHandler logic
│   ├── test_embed_handler.py            # EmbedHandler logic
│   ├── test_post_repository.py          # PostRepository.upsert, fetch
│   └── test_chunk_dispatcher.py         # ChunkDispatcher task routing
├── integration/
│   ├── test_chunk_repository.py         # Real Postgres, schema, indices
│   ├── test_gpu_worker.py               # GPU worker startup, auto-creation
│   └── test_post_sync_handler.py        # API sync handler with DB
└── e2e/
    ├── test_sync_api.py                 # POST /sync integration test
    └── test_ingest_pipeline.py          # Full pipeline: sync → chunk → embed
```

---

## Fixtures

### Database & Connection Pools

| Fixture | Scope | Provides |
|---------|-------|----------|
| `db_pool` | session | Real asyncpg Pool (test Postgres container) |
| `test_post_repo` | function | PostRepository instance |
| `test_chunk_repo` | function | ChunkRepository instance |
| `test_event_bus` | function | PostgresEventBus (for integration tests) |

### Mocked Dependencies

| Fixture | Scope | Provides |
|---------|-------|----------|
| `fake_post_fetcher` | function | In-memory PostFetcher (returns test data) |
| `fake_chunk_store` | function | In-memory ChunkStore (collects inserts) |
| `mock_embedding_model` | function | Deterministic mock SentenceTransformer |
| `mock_rmq_connection` | function | Mocked aio_pika.Connection |

### Test Data

| Fixture | Provides |
|---------|----------|
| `sample_post` | A valid Post instance (post_id=1, body_text="...") |
| `sample_chunks` | A list of 3 Chunk instances with different text_hashes |
| `sample_embedding_vector` | A deterministic embedding vector (dim=768) |

---

# Core Expected Behaviour Tests

List of user defined critical paths and behaviours that must be tested and not changed without a feature change.

## Sync Endpoint API

When I sync a new post it returns success and is added to the database and a inserted response 
When I sync an existing post with changes is updates and returns updated
When I sync an existing post with no changes it returns skipped

Post.synced event emitted if any changes. Event has skipped/inserted/updated and updated fields 

## Sync Event

Only dispatch a chunk post body if body or custom body updated.
Only dispatch chunk post summary if there is a summary and it's changed or if the title has changed

# CPU worker
Check that if a chunk task is processed, the correct embedding tasks are published.
Check that chunking tasks and search tasks are routed to the cpu worker, and embedding tasks are routed to the gpu worker.

# GPU Worker

Check there is a queue for each embedding model defined in config.
Check that search embedding are on a higher priority queue than chunk embedding tasks.
Check that if two tasks get added to a same queue they get batched and embedded together.
Check that if a task fails it gets requeued and eventually goes to the DLQ after max retries.

# Chunker
Check that if a post has no body, no chunk tasks are dispatched.
Check boundary logic for chunker

# Idempotency and deduplication
Check that if the same post is synced multiple times, it does not create duplicate chunks or embeddings by using post_updated_at from client as version and checking hashes before inserting new chunks or embeddings.

# Event Bus Publish and Consume
Events published to event bus are consumed by correct subscribers.
Consumer group offset is tracked and resumed correctly on restart

# Databases
First chunk task for a library+field+model triple creates the chunk table with correct vector dimensions
Re-syncing a post with updated chunks deduplicates via text_hash (existing chunks are skipped)
Switching embedding models creates a new chunk table without data loss

# API Error Handling
Invalid library_id in sync request is rejected with 400
Malformed post data (missing external_id, invalid updated_at) is rejected
DB connection loss during sync returns 500 (not crash)

# Dead-Letter Queue & Retry Exhaustion
Failed task is requeued per RabbitMQ retry policy (max delivery count)
After max retries, message lands in DLQ with original payload intact
DLQ messages can be inspected/replayed

# Task Dispatcher Event Filtering
Dispatcher does NOT publish chunk tasks if neither body nor custom_body changed (even if title changed)
Dispatcher does NOT publish summary chunk task if summary doesn't exist, OR if summary exists but didn't change (unless title changed!!)
 

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Start test Postgres container (if not using docker-compose)
docker run --rm -d \
  -e POSTGRES_DB=rag_test \
  -e POSTGRES_PASSWORD=rag \
  -p 5433:5432 \
  ankane/pgvector:latest
```

### Run All Tests

```bash
# Full test suite with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run integration tests (requires real DB)
pytest tests/integration/

# Run E2E tests (requires docker-compose)
docker-compose up -d
pytest tests/e2e/
docker-compose down
```

- Use `OTEL_ENABLED=true` for observability-enabled tests (e.g. GPU worker integration tests). 
- Use `-m e2e` to run only E2E tests.

### Run Specific Test

```bash
# Single test file
pytest tests/unit/test_chunk_handler.py

# Single test function
pytest tests/unit/test_chunk_handler.py::test_chunk_handler_splits_text

# Tests matching a pattern
pytest -k "test_chunk" --verbose
```

### With MOCK_EMBEDDINGS

For GPU worker tests without a real GPU:

```bash
MOCK_EMBEDDINGS=1 pytest tests/integration/test_gpu_worker.py -v
```

---

## Key Test Patterns

### Unit Test Pattern (Handler Logic)

```python
async def test_chunk_handler_splits_text(fake_post_fetcher, fake_chunk_store, sample_post):
    """Verify ChunkPostHandler chunks text and persists."""
    handler = ChunkPostHandler(
        post_fetcher=fake_post_fetcher,
        chunk_store=fake_chunk_store,
        version_checker=InMemoryVersionChecker({}),
        event_log=FakeEventBus(),
    )
    
    task = ChunkTask(post_id=1, post_table="posts", task_type="body")
    chunk_ids = await handler.handle(task)
    
    assert len(chunk_ids) == 3
    assert len(fake_chunk_store.chunks) == 3
```

### Integration Test Pattern (Real DB)

```python
async def test_chunk_repository_persists_embeddings(db_pool, sample_chunks):
    """Verify ChunkRepository.update_embeddings writes vectors."""
    repo = ChunkRepository(db_pool)
    await repo.ensure_table("chunks_body_bge", dim=768)
    
    await repo.bulk_insert(sample_chunks)
    embeddings = [[0.1] * 768 for _ in sample_chunks]
    await repo.update_embeddings("chunks_body_bge", sample_chunks[0].post_id, embeddings)
    
    rows = await repo.fetch_by_post_id("chunks_body_bge", post_id=1)
    assert all(row["embedding"] is not None for row in rows)
```

### E2E Test Pattern (API + Pipeline)

```python
async def test_sync_api_triggers_pipeline(client):
    """Verify POST /sync → chunks → embeddings (with polling)."""
    response = await client.post("/posts/sync", json={
        "posts": [{"id": 1, "title": "...", "bodyText": "..."}]
    })
    assert response.status_code == 200
    
    # Poll until chunks appear (timeout after 10s)
    for _ in range(20):
        chunks = await check_chunks_exist(post_id=1)
        if chunks:
            break
        await asyncio.sleep(0.5)
    
    assert len(chunks) > 0
```

---

## Mocking & Test Doubles

### Fake PostFetcher (in-memory)

```python
class FakePostFetcher:
    def __init__(self, posts_by_id: dict):
        self.posts_by_id = posts_by_id
    
    async def fetch(self, post_id: int) -> dict:
        return self.posts_by_id.get(post_id, {})
```

### Fake ChunkStore (in-memory)

```python
class FakeChunkStore:
    def __init__(self):
        self.chunks = []
    
    async def bulk_insert(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
```

### Mock Embedding Model (deterministic)

```python
class MockEmbeddingModel:
    def __init__(self, dim: int = 768):
        self.dim = dim
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [b / 255.0 for b in digest]
            vec = (vec * ((self.dim // len(vec)) + 1))[:self.dim]
            embeddings.append(vec)
        return embeddings
```

---

## Coverage Goals

- **Unit tests:** >90% coverage on handlers, repositories, dispatchers
- **Integration tests:** Verify DB operations and schema management
- **E2E tests:** Verify pipeline end-to-end (at least one happy path per flow)

Current coverage is tracked by pytest-cov. Run with `--cov-report=html` to see detailed breakdown.

---

## Known Test Limitations

1. **GPU Worker E2E:** Uses `MOCK_EMBEDDINGS=1` to avoid GPU requirement. Real model tests can be added with GPU CI.
2. **RabbitMQ Topology:** Tests mock RabbitMQ exchanges/queues. Full topology validation would require a real RabbitMQ instance.
3. **Consumer Groups:** Integration tests use Postgres event bus mock. Full consumer group behavior requires Redpanda.

---

## Debugging Tests

### Enable Debug Logging

```bash
pytest -v --log-cli-level=DEBUG tests/unit/test_chunk_handler.py
```

### Run Single Test with Print Output

```bash
pytest -s -v tests/unit/test_chunk_handler.py::test_my_test
```

### Drop into Debugger on Failure

```bash
pytest --pdb tests/unit/test_chunk_handler.py
```
