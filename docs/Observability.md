# Observability

This document explains how observability is designed in this system, and why the tradeoffs were made the way they were, not just what was implemented.

---

## What we're actually trying to see

The pipeline processes a single user action across five separate processes:

```
POST /posts/sync (API)
    ↓  Postgres event log
PostDispatcher  →  ChunkTask  →  CpuChunkWorker
    ↓  Postgres event log
ChunkDispatcher  →  EmbedTask  →  GpuEmbedWorker
```

Without observability, debugging means stitching together five separate log
streams and guessing which lines belong to the same request.  With traces,
you get a single waterfall view: one user action → every downstream span,
with timings.

---

## The Three Pillars

| Pillar | Question | Tool | Status |
|--------|----------|------|--------|
| **Logs** | What happened? | structlog → stdout → Alloy → Loki | ✅ Phase 1 + 4 |
| **Traces** | Where did time go? | OpenTelemetry → Tempo | ✅ Phase 2 |
| **Metrics** | Is the system healthy? | OTEL Metrics → Prometheus | ✅ Phase 3 |

These are not substitutes.  Logs are for humans debugging a specific incident.
Metrics are for alerting and dashboards.  Traces are for understanding latency
distribution across service boundaries.

Traces and metrics are visible in Grafana with correlation: click a Prometheus
exemplar to jump directly to the Tempo trace that produced it.  Click any log
line in Loki to jump to its Tempo trace via `trace_id`.

---

## Design Decisions and Their Reasons

### 1. Disabled by default

**The decision:** `OTEL_ENABLED=false` is the default.  No spans are exported,
no OTLP connections are opened, no background threads are started.

**The reason:** Running Jaeger + Prometheus + Grafana 24/7 on a homelab wastes
RAM and disk for zero benefit when nothing is broken.  The principle is
*"instrumentation is cheap, exporting is expensive"* — the code paths that
create spans always exist, but the export pipeline is off by default.

**What this means in practice:**  
When `OTEL_ENABLED=false`, the OTEL API returns `NonRecordingSpan` objects.
Every `with tracer.start_as_current_span(...)` block executes normally but
produces no telemetry data.  The overhead is measured in nanoseconds.

---

### 2. Trace context lives in the event schema, not transport headers

**The decision:** `trace_id` and `parent_span_id` are explicit fields on
`BaseEvent` and `BaseTask`, not hidden in Kafka headers or RabbitMQ properties.

**The reason:** Auto-instrumentation libraries for Kafka/RabbitMQ inject W3C
`traceparent` headers into the transport layer.  They see a Kafka produce call,
not your application's event structure.  This creates a hidden dependency on
the transport library and breaks the moment you change message brokers or add
a Postgres mock event bus.

Explicit schema fields mean:
- The trace context is visible and auditable in every event log row
- Switching from Redpanda to Postgres mock preserves full trace continuity
- No surprise breakage when transport libraries update

---

### 3. Both trace_id AND parent_span_id (not just trace_id)

**The decision:** Events and tasks carry `parent_span_id` in addition to
`trace_id`.

**The reason:** `trace_id` alone groups spans under the same trace in Jaeger,
but they appear as disconnected sibling spans at the same indentation level —
no waterfall.  `parent_span_id` (the `span_id` of the span that published the
event) tells the OTEL SDK "my parent is *this specific span*", creating the
arrow and the indented waterfall view.

Without `parent_span_id`:
```
trace abc123
  ├── sync_posts       (API, t=0ms)
  ├── dispatch_chunk   (Dispatcher, t=50ms)   ← no parent arrow
  └── chunk_post       (Worker, t=120ms)      ← no parent arrow
```

With `parent_span_id`:
```
trace abc123
  └── sync_posts       (API, t=0ms)
        └── dispatch_chunk  (Dispatcher, t=50ms)
                └── chunk_post  (Worker, t=120ms)
```

---

### 4. structlog over plain logging

**The decision:** All services use structlog to route stdlib `logging` calls
through a structured processor chain.

**The reason:** Python's stdlib logging produces unstructured text like:
```
2026-05-06 12:34:56 INFO chunk_handler: post 1 → 5 new chunks
```
Parsing that in a log aggregator requires fragile regex.  structlog renders
the same call as a JSON object:
```json
{"timestamp": "2026-05-06T12:34:56Z", "level": "info", "logger": "chunk_handler",
 "event": "post 1 → 5 new chunks", "trace_id": "4bf92f..."}
```

The `trace_id` appears automatically in every log record emitted inside a span
because of the `_inject_otel_context` structlog processor.  No log call sites
need changing — the correlation is ambient.

---

### 5. Instrument boundaries, not internals

**The decision:** Spans are created at service entry points only:
- `sync_posts` (API)
- `post_dispatcher.dispatch` / `chunk_dispatcher.dispatch`
- `chunk_post` (ChunkPostHandler)
- `embed_chunks` / `embed_query` (EmbedHandler)

**The reason:** Every span has a cost: creation, attribute storage, export.
More importantly, too many spans create noise that obscures real bottlenecks.
When you see 50 spans for one request, the 3-second outlier is buried.

The rule: **every span must answer a question**.  If you can't say "this span
tells me whether X is slow", delete it.

Repositories, helper functions, and loops are not instrumented.  If a DB query
is slow, it will show up as the `chunk_post` span being slow — which is enough
information to direct investigation.

---

### 6. BatchSpanProcessor in production, SimpleSpanProcessor in tests

**The decision:** Production uses `BatchSpanProcessor`; tests use
`SimpleSpanProcessor` with `InMemorySpanExporter`.

**The reason:**  
`SimpleSpanProcessor` exports synchronously on span end — it blocks the thread
and adds per-span I/O latency.  Fine for tests because we need spans available
immediately after the `with` block closes.

`BatchSpanProcessor` collects spans in memory and exports them on a background
thread in batches.  If the collector is slow or unreachable, spans queue up and
are retried — your handlers are never blocked waiting for telemetry I/O.  This
is the only acceptable choice for production.

---

### 7. Logs go to stdout — Grafana Alloy ships them to Loki

**The decision:** Logs are written to stdout only.  Grafana Alloy scrapes Docker
container stdout and pushes structured JSON to Loki.
OTLP log export is deliberately not used.

**The reason:** stdout is the universal logging contract for containers (the
12-factor app standard).  Keeping log shipping in a separate agent rather than
the OTEL pipeline has concrete operational advantages:

- **Crash resistance:** if the OTEL collector goes down, logs still land on
  stdout and remain readable via `docker compose logs`.  OTLP log export has no
  fallback — a collector outage silently drops records written nowhere else.
- **Separation of concerns:** the log backend (Loki → Elasticsearch → CloudWatch)
  can change without touching application code.
- **Simplicity:** structlog already writes structured JSON to stdout when
  `OTEL_ENABLED=true`.  No new SDK wiring, no new handler, no feedback-loop risk.

//
// WHY ALLOY for log shipping (vs OTLP log export from the app):
//   stdout is the universal container logging contract (12-factor).  If the
//   collector or Loki goes down, logs still land on stdout and are readable
//   via `docker compose logs`.  OTLP log export has no such fallback.
//   Alloy also adds label enrichment and pipeline processing outside the app.
//
// WHY ALLOY over Fluentd / Fluentbit / Logstash:
//   Alloy is Grafana's own agent (successor to Promtail + Grafana Agent Flow).
//   It speaks native Loki push format, requires no plugins for LGTM targets,
//   and its River HCL pipeline model is the same conceptual model as the rest
//   of the Grafana stack.  Fluentd/Fluentbit are vendor-neutral and need extra
//   config to reach Loki; Logstash is JVM-based and Elasticsearch-oriented.

**Why Grafana Alloy over Fluentd / Fluentbit / Logstash:**  
Alloy is Grafana's own collector agent — the successor to Promtail, Grafana
Agent, and Grafana Agent Flow, unified into a single binary.  It speaks native
Loki push format, has built-in OTEL receiver support, and its River HCL config
language is purpose-built for pipeline composition.

Fluentd and Fluentbit are excellent vendor-neutral shippers but require community
plugins and extra config to target Loki.  Logstash is JVM-based and optimised
for Elasticsearch.  For a Grafana homelab stack, Alloy is the natural choice —
maintained by the same team as Loki and Grafana.

---

## Trace flow in this pipeline

One HTTP request creates one root span.  Its `trace_id` and `span_id` flow
through every downstream event and task:

```
sync_posts                          ← root span (API)
  │  trace_id: abc123, span_id: 001
  ↓  PostSyncedEvent { trace_id: abc123, parent_span_id: 001 }
post_dispatcher.dispatch            ← child span (reads parent_span_id=001)
  │  trace_id: abc123, span_id: 002
  ↓  ChunkTask { trace_id: abc123, parent_span_id: 002 }
chunk_post                          ← child span (reads parent_span_id=002)
  │  trace_id: abc123, span_id: 003
  ↓  ChunksCreatedEvent { trace_id: abc123, parent_span_id: 003 }
chunk_dispatcher.dispatch           ← child span (reads parent_span_id=003)
  │  trace_id: abc123, span_id: 004
  ↓  EmbedTask { trace_id: abc123, parent_span_id: 004 }
embed_chunks                        ← child span (reads parent_span_id=004)
```

Each span knows its parent because we pass the `parent_span_id` from the
previous span.  This is what `extract_trace_context(trace_id, parent_span_id)`
reconstructs — a `NonRecordingSpan` that carries the IDs without itself being
exported, used only to set the `parentSpanId` field on the new real span.

---

## Span attributes (what to look for in Jaeger)

| Span | Key attributes |
|------|----------------|
| `sync_posts` | `library_id`, `post_count`, `inserted_count`, `skipped_count` |
| `post_dispatcher.dispatch` | `post_id`, `post_table` |
| `chunk_dispatcher.dispatch` | `post_id`, `chunk_count`, `model` |
| `chunk_post` | `post_id`, `task_type`, `total_chunks`, `new_chunks`, `skipped_chunks` |
| `embed_chunks` | `model`, `task_count` |
| `embed_query` | `model`, `query_job_id` |

---

## Key files

| File | Purpose |
|------|---------|
| [`utils/tracing.py`](../src/event_driven_rag_service/utils/tracing.py) | `traced()`, `extract_trace_context()`, `current_trace_ids()` |
| [`infrastructure/observability.py`](../src/event_driven_rag_service/infrastructure/observability.py) | `setup_observability()` — structlog + OTEL SDK wiring |
| [`events/base_event.py`](../src/event_driven_rag_service/events/base_event.py) | `trace_id` + `parent_span_id` on all events |
| [`tasks/base_task.py`](../src/event_driven_rag_service/tasks/base_task.py) | `trace_id` + `parent_span_id` on all tasks |
| [`tests/unit/test_trace_propagation.py`](../tests/unit/test_trace_propagation.py) | In-memory exporter tests for the tracing utilities |

---

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OTEL_ENABLED` | `false` | `true` activates OTLP export + JSON logs |
| `OTEL_SERVICE_NAME` | `rag-pipeline` | Service label in Tempo/Grafana (set per-container in docker-compose) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC collector endpoint |

---

## Enabling observability

```bash
# Default: pipeline only — no observability containers started
docker compose up

# Enable: pipeline + full LGTM stack (6 extra containers)
OTEL_ENABLED=true docker compose --profile observability up
```

The `observability` Docker Compose profile gates all six observability containers
(otel-collector, alloy, tempo, loki, prometheus, grafana).  A plain
`docker compose up` runs the pipeline with zero observability overhead — no
collector, no Grafana, no background exporters.

With `OTEL_ENABLED=true --profile observability`:

- Grafana (traces + metrics + logs): `http://localhost:3000` (admin/admin)
- Prometheus (metrics): `http://localhost:9090`
- Tempo HTTP API (traces): `http://localhost:3200`
- Loki HTTP API (logs): `http://localhost:3100`

---

## Phase 3: Metrics — Turning traces into system health

Traces answer "why is this slow?" — they're tools for incident response.
Metrics answer "is the system degrading?" — they drive dashboards and alerting.

### Golden metrics: the minimum viable observability

The system exports OTEL metrics (not Prometheus client library directly).
We define only metrics that answer real operational questions:

| Metric | Question | Type |
|--------|----------|------|
| `posts_processed_total` | How much throughput? | Counter |
| `chunks_created_total` | How many chunks? | Counter |
| `chunks_deduplicated_total` | Idempotency working? | Counter |
| `embeddings_generated_total` | GPU output? | Counter |
| `failures_total` | Where are errors? | Counter |
| `pipeline_latency_seconds` | API e2e time? | Histogram |
| `chunking_latency_seconds` | CPU worker speed? | Histogram |
| `embedding_latency_seconds` | GPU worker speed? | Histogram |

### Design principle: low cardinality labels only

Each metric has a few labels: `status`, `task_type`, `model`, `service`.
**Never** add user_id, post_id, or other unbounded fields — they cause
Prometheus cardinality explosion and memory exhaustion.

### Where metrics are recorded

- **API** (`api/sync.py`): posts_processed_total, pipeline_latency_seconds, failures_total
- **Chunk handler** (`handlers/chunk_handler.py`): chunks_created_total, chunks_deduplicated_total, chunking_latency_seconds
- **Embed handler** (`handlers/embed_handler.py`): embeddings_generated_total, embedding_latency_seconds, failures_total

---

## Phase 4: Observability stack — LGTM (Loki + Grafana + Tempo + Prometheus)

**Now implemented.** To use:

```bash
OTEL_ENABLED=true docker compose up
```

The stack provides:

- **OTEL Collector**: Receives OTLP gRPC from services, routes traces + metrics
- **Grafana Alloy**: Scrapes Docker container stdout, parses JSON, pushes to Loki
- **Tempo**: Stores traces; queried by Grafana at `http://localhost:3200`
- **Loki**: Stores logs (shipped by Alloy); queried by Grafana at `http://localhost:3100`
- **Prometheus**: Scrapes metrics from the OTEL Collector exporter (`http://localhost:9090`)
- **Grafana**: All three pillars in one UI (`http://localhost:3000`, admin/admin)

### How the data flow works

```
structlog / logging.getLogger()
  ↓ stdout (always)                       ↓ OTLP gRPC (OTEL_ENABLED=true)
Grafana Alloy → Loki                 OTEL Collector
  (scrapes Docker stdout)              ├→ Traces  → Tempo
                                       └→ Metrics → Prometheus
                                             ↓
                                         Grafana (datasources: Loki, Tempo, Prometheus)
```

Grafana datasource correlation is fully wired:
- Tempo trace span → "View Logs" jumps to Loki filtered by `trace_id`
- Loki log line → "View Trace" opens the matching Tempo trace
- Prometheus exemplar → links directly to its Tempo trace

When `OTEL_ENABLED=false`, all OTLP export is disabled but instrumentation
remains cheap — spans are created as `NonRecordingSpan` objects with zero overhead.

---

## What NOT to instrument

This system deliberately does not span:
- Repository methods (`PostRepository`, `ChunkRepository`)
- Chunking strategy functions
- Data validation helpers
- Loops inside handlers

If a repository query is slow, it shows as a long `chunk_post` span.  That's
the right signal — it directs you to the handler boundary, which is where you
can act (query optimization, index tuning).  Adding a per-query span would add
noise without adding actionable information at this stage.

---

## Completed phases

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | structlog + OTEL SDK wiring | ✅ Done |
| **2** | Trace propagation across all service boundaries | ✅ Done |
| **3** | Golden metrics: counters, latency histograms, low-cardinality labels | ✅ Done |
| **4** | Observability stack: LGTM (Loki + Grafana + Tempo + Prometheus) with trace↔log↔metric correlation | ✅ Done |

---

## Future phases

| Phase | Focus | Status |
|-------|-------|--------|
| **5** | Sampling strategy (reduce trace volume at scale) | Planned |
| **6** | DLQ observability (dead letter queue metrics + alerts) | Planned |
| **7** | SLO definitions and alerting rules | Planned |
