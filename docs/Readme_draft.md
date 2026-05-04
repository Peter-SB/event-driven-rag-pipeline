# Readme Draft

# Event Driven RAG Pipeline


## Tech Stack

FastAPI 
Postgres + PGvector

Redpanda (Postgress mock) - Events log, lightweight Kafka compatible
RabbitMQ - Task queue message broker

K3S - Lightweight Kubernetes distribution

Prometheus + Graphana for observability


# Arcetecutre 

## Two Layers: Event and Tasks
```
┌─────────────────────────────────────────────────────┐
│  EVENT LOG (Redpanda/Kafka)                          │
│  Immutable, replayable, ordered. Source of truth.   │
│  "What happened"                                     │
└─────────────────────┬───────────────────────────────┘
                      │ consumers derive work from events
┌─────────────────────▼───────────────────────────────┐
│  TASK QUEUE (RabbitMQ)                               │
│  Mutable, ack-based, routed to worker pools.         │
│  "What needs to be done"                             │
└─────────────────────────────────────────────────────┘
```

solve different problems. Redpanda is your audit log and integration backbone. RabbitMQ is your work distribution mechanism with routing, priorities, and backpressure.

### Event Flow Design
```
post.synced (Redpanda)
      │
      ├── Ingestion Consumer → publishes tasks to RabbitMQ
      │         exchange: ingestion
      │              ├── queue: chunk          → CPU workers
      │              └── queue: inference      → GPU workers
      │
      └── (other future consumers e.g. analytics, audit)

embedding.batch.ready (Redpanda)
      │
      └── Embedding Consumer → RabbitMQ
               exchange: embedding
                    └── queue: embed.batch     → GPU workers

search.job.created (Redpanda)
      │
      └── Search Consumer → RabbitMQ
               exchange: search
                    ├── queue: embed.query     → GPU workers
                    └── queue: search.execute  → CPU workers

WORLD                EVENT LOG              DISPATCHER          TASK BROKER
                     (Redpanda)                                  (RabbitMQ)

Something       →    "It happened"     →    reads event    →    "Go do X"
happened             immutable              decides work         routable
                     replayable             fan-out              ack-based
```

**Each Component, One Sentence** (Definitions)
Producer — anything that says "something happened" by writing to Redpanda.
Event — an immutable fact. Past tense. Nobody tells it where to go.
Consumer — anything that reads from Redpanda. Could be a dispatcher, analytics service, audit logger — anything.
Dispatcher — a specific type of consumer. Reads events, translates them into tasks, publishes to RabbitMQ. It's the bridge. It's the only thing that knows about both layers.
Exchange — RabbitMQ's routing mechanism. It receives a message and decides which queues get it, based on the routing key. It is not a queue itself.
Queue — where tasks actually sit and wait for a worker.
Task — a unit of work. Imperative. Present tense. Has a specific destination.
Worker — consumes from a RabbitMQ queue. Does the actual work. Knows nothing about Redpanda.

**Concrete Mapping to Your Pipeline**

```
┌─────────────────────────────────────────────────────────────────────┐
│ PRODUCERS (write events to Kafka/Redpanda)                                │
│                                                                     │
│  • Your API / sync service    → post.synced                         │
│  • ChunkWorker (after done)   → post.chunked                        │
│  • EmbedWorker (after done)   → embedding.completed                 │
│  • SearchAPI                  → search.job.created                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ events flow into Redpanda topics
┌────────────────────────────▼────────────────────────────────────────┐
│ REDPANDA TOPICS (event log)                                         │
│                                                                     │
│  post.synced          → { post_id, has_summary, source }            │
│  post.chunked         → { post_id, chunk_ids }                      │
│  embedding.completed  → { post_id, chunk_ids }                      │
│  search.job.created   → { job_id, query, filters }                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ consumers read from topics
┌────────────────────────────▼────────────────────────────────────────┐
│ DISPATCHERS (Redpanda consumers → RabbitMQ publishers)              │
│                                                                     │
│  PostDispatcher                                                     │
│    reads:    post.synced                                            │
│    publishes tasks to RabbitMQ:                                     │
│      ingestion exchange  →  routing_key: post.chunk.body            │
│      inference exchange  →  routing_key: post.summarise             │
│      inference exchange  →  routing_key: post.categorise            │
│                                                                     │
│  EmbeddingDispatcher                                                │
│    reads:    post.chunked                                           │
│    publishes tasks to RabbitMQ:                                     │
│      embedding exchange  →  routing_key: batch.chunks               │
│                                                                     │
│  SearchDispatcher                                                   │
│    reads:    search.job.created                                     │
│    publishes tasks to RabbitMQ:                                     │
│      embedding exchange  →  routing_key: batch.query                │
│      search exchange     →  routing_key: job.execute                │
└────────────────────────────┬────────────────────────────────────────┘
                             │ tasks flow into RabbitMQ
┌────────────────────────────▼────────────────────────────────────────┐
│ RABBITMQ EXCHANGES + QUEUES                                         │
│                                                                     │
│  exchange: ingestion  (topic)                                       │
│    post.chunk.*    →  q.chunk                                       │
│    post.infer.*    →  q.inference                                   │
│                                                                     │
│  exchange: embedding  (topic)                                       │
│    batch.chunks    →  q.embed.chunks                                │
│    batch.query     →  q.embed.query                                 │
│    batch.summary   →  q.embed.summary                               │
│                                                                     │
│  exchange: inference  (topic)                                       │
│    post.summarise  →  q.summarise                                   │
│    post.categorise →  q.categorise                                  │
│                                                                     │
│  exchange: search  (topic)                                          │
│    job.execute.*   →  q.search.execute                              │
│    job.aggregate   →  q.search.aggregate                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ workers pull from queues
┌────────────────────────────▼────────────────────────────────────────┐
│ WORKERS (consume RabbitMQ queues, do work, produce new events)      │
│                                                                     │
│  CPU Machine                                                        │
│    ChunkWorker       ←  q.chunk                                     │
│    SearchWorker      ←  q.search.execute                            │
│    AggregateWorker   ←  q.search.aggregate                          │
│                                                                     │
│  GPU Machine                                                        │
│    EmbedWorker       ←  q.embed.chunks, q.embed.query               │
│    SummariseWorker   ←  q.summarise                                 │
│    CategoriseWorker  ←  q.categorise                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

rag_pipeline/
├── infrastructure/
│   ├── rabbitmq.py             # topology setup, connection factory
│   ├── redpanda.py             # producer/consumer base classes
│   └── sqlite_mock.py          # drop-in mock for homelab
│
├── events/                     # Redpanda layer — event definitions
│   ├── base.py
│   ├── post_events.py
│   └── search_events.py
│
├── workers/                    # RabbitMQ consumers — do actual work
│   ├── base_worker.py
│   ├── embed_worker.py         # GPU machine
│   ├── inference_worker.py     # GPU machine
│   ├── cpu_worker.py
│   └── search_worker.py
│
├── dispatchers/                # bridge: Redpanda event → RabbitMQ task
│   ├── post_dispatcher.py
│   └── search_dispatcher.py

Producers, consumers??


### Layers

Event Layer (Kafka)
- Durable, replayable
- “What happened in the system?”

Task Layer (RabbitMQ)
- Ephemeral execution
- “What needs to be done right now?”

Database Layer (Postgres + pgvector)
- Post repo
- chunk repo (for all chunked data that )
- search job repo

# Core constraints 

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






# Core design decsisions

- Redpada for a kafka compatible demo and learning, sqlite mock for homelab and integration testing
- Embedding and inference worker is built to minimise model load times, they poll resource specific queues such as embedding.{model} in a specific order or priority.
- tasks should be pure and easily testable

1. Redpanda (Kafka) as Event Backbone
Enables replay (critical for RAG reprocessing)
Decouples producers from consumers
Supports multiple independent pipelines:
ingestion
analytics
audit
2. RabbitMQ as Execution Layer
Handles:
routing
retries
prioritization
Better suited than Kafka for task distribution
3. No Celery — Custom Workers

Celery is intentionally avoided due to:

Poor batching support
No control over model lifecycle
Inefficient GPU usage

Instead, we use:

long-lived workers
explicit batching
manual polling strategies
4. Postgres Event Bus (Homelab Mode)

A drop-in Kafka replacement:

Append-only log
Consumer offsets
Replay support
Zero infrastructure

