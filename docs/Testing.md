# Tests

This document covers the test strategy, structure, naming conventions, and how to run each suite. Testing is treated as a first-class concern in this project — a distributed pipeline with GPU workers, event buses, and task queues has a lot of moving parts, and tests are the primary defence against regressions.

---

## Philosophy

> Test behaviour, not infrastructure.

Tests should verify that the system does what it is supposed to do, not that it calls specific functions in a specific order. A test that just restates the implementation provides no value and breaks on every refactor.

Tests are also documentation. A well-named test with a brief comment explains the intent of the code being tested and the criteria for success. Future contributors (including yourself) should be able to read the test suite and understand the system's expected behaviour without reading the implementation.

---

## Test Categories

The suite is split into three tiers with different scope, speed, and infrastructure requirements.

### Unit Tests

**Scope:** Pure logic, no external dependencies.

**Infrastructure:** None. No database, no containers, no network. All dependencies are mocked or faked.

**When to use:** For testing the core logic of individual components in isolation. Fast enough to run on every save. The sanity check layer.

**What belongs here:**
- Event dataclass construction and serialisation
- Dispatcher routing logic (given this event, these tasks are published)
- Worker processing logic with a fake queue
- Repository query construction
- Utility functions

**What does not belong here:** Anything that requires a real connection to verify correctness. If you need a real DB to know if the query works, that is an integration test.

---

### Integration Tests

**Scope:** Interaction between components with real infrastructure.

**Infrastructure:** Real Postgres and RabbitMQ and redpanda via `testcontainers`. Containers spin up for the test run and are torn down afterwards. No manual setup required. Use redpanda for testing the event bus integration but use mock event bus for tests where we want to easily inspect the published events table without having deal with consumer groups and offsets.

**When to use:** Where unit tests with mocks cannot verify the actual behaviour — SQL queries against real data, RabbitMQ message routing, offset tracking in the Postgres event bus mock.

**What belongs here:**
- Repository operations against real Postgres (insert, query, pgvector search)
- RabbitMQ exchange + queue topology verification
- Event bus publish → consume roundtrip (Postgres mock)
- Dispatcher end-to-end (event in → tasks out, with real queue)
- Consumer offset tracking and replay behaviour

**What does not belong here:** Full pipeline tests that require all services running. That is E2E.

---

### End-to-End (E2E) Tests

**Scope:** The full system in a production-like environment.

**Infrastructure:** Full Docker Compose stack. All services running, like production.

**When to use:** Critical user journeys only. These tests are slow and expensive to maintain. Keep the count low but meaningful.

**What belongs here:**
- Post sync → chunks stored → embeddings stored (full ingestion path)
- Search request → results returned (full search path)
- Failure and retry behaviour (worker fails, message goes to DLQ, reprocessed)
- Replay scenario (consumer group reset, events reprocessed correctly)

**Target:** Enough tests to be confident the system works end-to-end. Not so many that the suite takes 20 minutes or becomes fragile.

---

## Naming Conventions

### Test function names

Follow standard pytest naming: `test_[functionality]_[expected_result]` with an optional `_when_[condition]` suffix for specificity.

```
test_post_dispatcher_publishes_chunk_task_on_post_synced
test_embed_worker_skips_already_embedded_chunks
test_search_returns_unique_chunk_per_post_when_unique_true
test_consumer_offset_advances_after_successful_processing
test_dlq_receives_message_when_worker_raises
```

Avoid: `test_dispatcher`, `test_it_works`, `test_1`. Names should be self-describing.

### Test file names

Match the module being tested: `test_post_dispatcher.py`, `test_embed_worker.py`, `test_vector_repo.py`.

---

## Documentation Conventions

### File header

Every test file begins with a brief docstring covering:
- What is being tested
- Any important context or setup
- List of the main behaviours being verified

```python
"""
Tests for PostDispatcher.

PostDispatcher reads post.synced events from the event bus and publishes
tasks to RabbitMQ ingestion and inference exchanges. These tests verify
routing logic using a fake event bus and a mock RabbitMQ publisher.

Tested behaviours:
- chunk task always published on post.synced
- summarise task published only when has_summary=True
- categorise task always published
- malformed events are logged and skipped, not raised
"""
```

### Test comments

Use comments to explain intent, not mechanics. Avoid restating what the code does.

```python
# Good — explains why, not what
# Summarise task should only be dispatched when the post already has a summary
# to avoid inference on posts that will never need it
assert summarise_task not in published_tasks

# Bad — restates the code
# Assert that summarise_task is not in published_tasks
assert summarise_task not in published_tasks
```

Comments should be skimmable. A few lines of context, not an essay.

---

## Running Tests

### Unit tests

```bash
pytest tests/unit/
```

No setup required. Runs in seconds.

### Integration tests

```bash
pytest tests/integration/
```

Requires Docker. `testcontainers` pulls and starts a Postgres image automatically — no manual setup. First run is slower due to the image pull; subsequent runs reuse the cached image.

### E2E tests

```bash
docker compose up -d
pytest tests/e2e/ -m e2e
docker compose down
```

The full Docker Compose stack must be running before E2E tests are invoked. Tests connect to the real API server via HTTP (not ASGI transport) and query Postgres directly to verify pipeline state. CI handles stack startup automatically.

### All tests

```bash
pytest
```

---

## Key Fixtures

Each tier has its own `conftest.py`. Fixtures do not bleed across tiers.

### Unit (`tests/unit/conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `fake_bus` | function | In-memory `FakeEventBus`. Records published events; yields them on subscribe. No I/O. |
| `fake_exchange` | function | In-memory `FakeExchange`. Records `publish()` calls with routing keys. |

### Integration (`tests/integration/conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `postgres_container` | session | Postgres testcontainer (ankane/pgvector). Pulled once, shared across the integration session. |
| `postgres_dsn` | session | Raw asyncpg-compatible DSN from the container. |
| `postgres_pool` | function | Fresh asyncpg pool per test, bound to the test event loop. |
| `clean_event_bus_tables` | function | Drops and recreates `event_log` + `consumer_offsets` before each test. |
| `clean_posts_table` | function | Creates (or truncates) `test_posts` and yields a bound `PostRepository`. |
| `clean_chunk_table` | function | Creates (or truncates) a test chunk table and yields a bound `ChunkRepository`. |

### E2E (`tests/e2e/conftest.py`)

| Fixture | Scope | Description |
| --- | --- | --- |
| `postgres_pool_e2e` | function | asyncpg pool connected to the Docker Compose Postgres instance (via `DB_URL`). |
| `rmq_connection_e2e` | function | aio_pika connection to the Docker Compose RabbitMQ instance (via `RABBITMQ_URL`). |
| `async_client` | function | httpx `AsyncClient` pointed at the running API server (via `API_BASE`). Runs pre-test cleanup of e2e tables. |

---

## Updating Tests

When adding a feature or fixing a bug:

- Add or update tests to cover the new code path
- Verify the test fails without the fix, then passes with it
- Avoid tests that simply restate the implementation — test the outcome, not the mechanics
- If a bug was caught in prod, add a regression test that would have caught it

A test that does not fail when the thing it is testing is broken is not a test.

---

## CI

Unit and integration tests run on every push. E2E tests run on merge to `main`. The split keeps the fast feedback loop fast while still covering the full system before anything ships.

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
 