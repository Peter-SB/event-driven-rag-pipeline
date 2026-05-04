# Readme

# Event-Driven Cloud-Sync and RAG Pipeline

This project is a two-layer event-driven pipeline for backing up/syncing, embedding, analysing, and searching documents and content.

This is both a careful design choice and a learning exercise. The two layers provide a opportunity to explore these two distinct but complementary patterns for building distributed systems: **Event Log** and **Message Queues**. It also allows us to demonstrate how to build a production-inspired event-driven architecture for RAG pipelines in distributed systems at scale, while also being flexible, modular and lightweight, runnable on a homelab setup.  

The two layers are kept deliberately separate:

- **Event Log** (Kafka/redpanda/Postgres mock) — immutable, replayable record of what happened
- **Message Task Queue** (RabbitMQ) — ephemeral, routable work distributed to worker pools



# Tech Stack

| Layer         | Production                | Homelab / Dev |
|---            |---                        |---|
| Event Log     | Redpanda (Kafka-compatible) | Postgres mock |
| Task Queue    | RabbitMQ                  | RabbitMQ |
| Database      | Postgres + pgvector       | Postgres + pgvector |
| API           | FastAPI                       | FastAPI |
| Orchestration | K3s (lightweight Kubernetes) | Docker Compose |
| Observability | Prometheus + Grafana + OpenTelemetry | Optional / debug mode |

## Architecture at a Glance

```
API / Sync Service
       │
       ▼
Event Log (redpanda)          ← immutable, replayable, source of truth
       │
  Dispatchers                 ← read events, publish tasks (bridge layer)
       │
       ▼
Task Queue (RabbitMQ)         ← routing, retries, backpressure, DLQ
       │
  ┌────┴────┐
  ▼         ▼
CPU       GPU
Workers   Workers             ← do actual work, emit new events
  │
  ▼
Postgres + pgvector           ← persisted state
```

For a more detailed breakdown of exchanges, queues, topics, and worker routing see [Architecture.md](./Architecture.md).

## System and Design Constraints

This project was designed to demonstrate a production-ready event-driven architecture for RAG pipelines in distributed systems at scale. However I also intend to use this on my homelab, so the system is design to be flexible and modular, allowing for different infrastructure choices (e.g Redpanda vs Postgres mock) without changing the core architecture or design patterns. The core constraints are:
1. Minimal idle resource usage (e.g no expensive infrastructure for homelab)
2. Efficient GPU usage (e.g model load strategies, batching)
3. Clear separation of concerns (e.g producers, consumers, dispatchers)
4. Observability (logs, metrics, tracing)
5. Testability (unit, integration, end-to-end)
6. Extensibility (e.g adding new event types, new consumers, new pipelines should be easy and not require changes to existing code)

## GPU Worker Design Constraints

GPU workers are designed to minimize model load times and maximize throughput by embedding in batches. Queues are organised by resource type (e.g embedding.{model}) and workers poll in a specific order to prioritise critical tasks. By grouping tasks by model and using long-lived workers, we can keep models warm and reduce latency. Only swapping or unloading models when the model required changes or there are no more tasks for that model. Minimising load times and unloading models only when necessary is key to efficient GPU usage for this project.

# Core Constraints and Design Principles

**Minimal Idle Resource Usage:** Though design to be scalable and production-ready, the system have the practical constraint of needing to be runnable on a homelab setup without expensive idle infrastructure. A lightweight Postgres interface is used to mock the event log for homelab/dev environments while redpanda is used for production as a lightweight Kafka-compatible event log. The observability stack is design to be optional for homelab/dev environments.

**GPU and Model Load Optimisations:** GPU workers are designed to minimise model load times and maximise throughput by smart model loading strategy and embedding in batches. Queues are organised by resource type (e.g embedding.{model}) and workers poll in a specific order to prioritise critical tasks. By grouping tasks by model and using long-lived workers, we can keep models warm and reduce unnecessary model swapping. Only swapping or unloading models when the model required changes or there are no more tasks for that model. Polling queues and embedding in batches also optimises GPU usage for embedding.

**Clear Separation of Concerns:** This project is a redesign of my previous RAG pipeline project which grew confusing as tightly coupled tasks were added. By implementing a clear separation of concerns between producers, consumers, and dispatchers, we can keep components decoupled and focused on their specific responsibilities. Producers emit events without knowledge of who consumes them. Consumers listen for and react to events without knowledge of who produced them. Dispatchers bridge the event log and task queue, translating events into tasks without doing any actual work themselves. This separation allows for better modularity, testability, and extensibility.

**Observability:** Observability is a core design principle for this project. The system is designed to be observable with logs, metrics, and tracing to provide visibility into the behavior of the system, measure performance, and help with debugging. The observability stack is optional for homelab/dev environments to keep resource usage low, but can be enabled for production or when needed.

**Testability:** A good test strategy is critical, especially for a complicated distributed system. Testing code should be treated with the same care and attention as production code. Tests should focus on multiple layers: unit tests verify isolated task logic with mocked dependencies, integration tests verify component interaction with real infrastructure, and end-to-end tests verify the complete pipeline. Tests should be well-structured, meaningful, and maintainable. They should provide confidence that the system works as expected and help catch regressions early. Tests should also be documented to explain their purpose, criteria for success, and any important context or setup.

**Reliable and Scalable Architecture:** The architecture is designed to be reliable and scalable, with a focus patterns used in production systems. By using an event-driven architecture with a clear separation of concerns, we can build a system that is resilient to failures, can handle increasing loads, and can be extended with new features or components without requiring changes to existing code.




# Pipeline

### Post synced example:

A post comes in via the FastApi endpoint and is saved to the postgres database. An event is emitted "post.synced" with the post_id. Then a dispatpatcher listens for "post.synced" events and when it sees one, it publishes tasks to:
1. chunk the post body and save chunks to the database
2. if the post has a summary, embed the title and summary
3. generate an analysis of the post using an inference task

### Search Example:
A search job is created via the API and is saved to the database. An event is emitted `search_job.created` with the job_id and query. `SearchDispatcher` listens for `search_job.created` and queues the query embedding task. Once the GPU worker embeds the query, it emits `search_query.embedded`. `SearchDispatcher` listens for that event and publishes a `cpu.search.run` task to execute the search.

### Notes:

- Post is synced. Event emitted "post.synced"
	- Body is chunked and embedded 
	- If Summary, embed title+summary
	- Other tasks such as inference and catagorsation E.g Gen summary/analysis  

- Embedding tasks embed batches and save results in chunk table
	- .
- Search will save a search job, 
	- Embed queries 
		- Run and aggregate results
- Hypothetical research agentic endpoint. 
	- plan search strategy 
		- call tools such as Reddit API to do searches
		- check own database 
			- call LLM on results to summarise for example
				- feed results back
					- compile top candidate list


# Event to Task Mapping

| Event | Dispatcher | Tasks dispatched |
|---|---|---|
| `post.synced` | `PostDispatcher` | `cpu.chunk.post` (body) · `cpu.chunk.post` (summary, if present) · `gpu.infer_local.{model}` (analysis) |
| `post.analysed` | `PostDispatcher` | `cpu.chunk.post` (analysis) |
| `post.chunked` | `ChunkDispatcher` | `gpu.embed.{model}` |
| `chunk.created` | `ChunkDispatcher` | `gpu.embed.{model}` |
| `search_job.created` | `SearchDispatcher` | `gpu.embed.{model}` (query) |
| `search_query.embedded` | `SearchDispatcher` | `cpu.search.run` |




# Table Naming Conventions

## Database  
post table: `posts_{id}` where id come from the client and is the key for that library. E.g `posts_main`, `posts_work`, `posts_test`. This always comes from client
chunk table: `posts_{id}.chunks.{data_type}.{model}` e.g `posts_main.chunks.body.qwen3-06B`
embeddings are tightly coupled to chunks, so stored in the same table with a pgvector column for the embedding vector.
search job table: `posts_{id}.search_jobs`

## Events
`{entity}.{past_tense_verb}` e.g `post.synced`, `chunk.created`, `embedding.completed`, `search_query.embedded`

## Tasks
`{resource}.{action}.{qualifier?}` e.g `chunk.post`, `embed.query.{model}`, `search.run`




# Tests

A good test strategy is core to a complicated distributed system. Testing code should be treated with the same care and attention as production code. Tests should be well-structured, meaningful, and maintainable. They should provide confidence that the system works as expected and help catch regressions early. Tests should also be documented to explain their purpose, criteria for success, and any important context or setup.

### Test Structure

The test suite is broken into three categories.

**Unit Tests:** Pure Unit Tests (no DB, no containers). Testing units of code without external dependencies and side effects. Fast, zero-infrastructure, sanity check to run on every save. Uses mocks/fakes for all dependencies. Focus on testing the core logic of individual components in isolation and sanity checks for common cases. Dependency injection should be used heavily to enable mocking and isolation of units under test. Use these tests for all core logic that can be meaningfully tested in isolation without real dependencies.

**Integration Tests:** Test the interaction between components with a real database and queue, using testcontainers. Focus on testing the integration of components and the behavior of the system which cannot be tested by unit tests alone. With real dependencies, but without the overhead of full end-to-end testing. Use these tests where unit test arent sufficient to verify the behaviors of the system with real dependencies, but where full end-to-end testing would be too slow or complex.

**End-to-End (E2E) Tests:** Full stack tests running against the entire Docker Compose environment. Focus on testing the system as a whole end-to-end, including all components and their interactions in a production-like environment. Few tests that cover the most critical user journeys and edge cases, to verify that the system works as expected from end to end. Not so many tests that running the full suite becomes a burden, but enough to provide confidence that the system works as expected in a production-like environment.

### Updating Tests

After implementing a new feature or fixing a bug, add or update tests to cover the new code paths and verify the fix. Ensure the tests are meaningful and validate the expected behavior. Avoid adding tests that simply restate the implementation without verifying the intended outcomes or edge cases.

### Test Documenting Conventions

Give tests helpful names and comments to explain their purpose and criteria.

**Test Naming:** Use common pytest nameing practice: test functionality expected result (and condition if very specific). `test_[functionality]_[expected_result]_when_[condition]`. e.g `test_search_unique_true_returns_one_chunk_per_post`

**Test Comments:** Use comments to explain the purpose of the test, the criteria for success, and any non-obvious setup or assertions. Avoid restating what the code does; focus on the why, the criteria for success, and the intent behind the test. Dont be overly verbose, should be skimmable but informative.

**Test File Documenting:** At the top of the test file include a breif description of the overall purpose of the tests in that file, any important context or setup, and a list of the main functionalities being tested.

### Test Suite Documenting

TESTS.md should include a high-level overview of the test suite structure, instructions for running the tests, and any important context about the testing approach or architecture. It should also include a directory structure of the tests with brief descriptions of what each test file covers.
