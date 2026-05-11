"""
Observability bootstrap for all pipeline services.

Implements the "three pillars" (logs, metrics, traces) using OpenTelemetry + structlog.

--------------------------------------------------------------------------------
DESIGN PRINCIPLE: "Instrumentation is cheap. Exporting is expensive."
--------------------------------------------------------------------------------
The OTEL SDK and structlog are always configured at process startup.  But the
OTLP exporter (which opens a gRPC connection and sends data over the network)
only starts when OTEL_ENABLED=true.

When OTEL_ENABLED=false (the default):
  - structlog renders human-readable, coloured console output
  - The OTEL API returns no-op tracers and spans  →  zero CPU/memory overhead
  - No network connections, no background threads, no collector required

When OTEL_ENABLED=true:
  - structlog renders JSON so log aggregators (Loki) can parse it
  - A real TracerProvider is registered globally with a BatchSpanProcessor
  - Every log record gains trace_id + span_id from the active OTEL span
  - Traces and metrics export via OTLP gRPC to the configured collector endpoint

--------------------------------------------------------------------------------
LOG APPROACH: stdout → Grafana Alloy → Loki
--------------------------------------------------------------------------------
Logs are written to stdout only.  Grafana Alloy (the log shipper) scrapes Docker
container stdout and pushes structured JSON to Loki.  This is the standard,
battle-tested container logging pattern and has several advantages over shipping
logs through the OTEL pipeline:

  Crash resistance: if the OTEL collector goes down, logs still land on stdout
    and remain readable via `docker compose logs`.  OTLP log export has no
    fallback — a collector outage silently drops records that were never
    written anywhere else.

  Simplicity: zero application code changes.  structlog already writes structured
    JSON to stdout.  The agent handles discovery, buffering, retries, and label
    enrichment outside the application process.

  Standard: stdout/stderr is the universal logging contract for containers (the
    12-factor app standard).  Every log aggregation system supports it.  OTLP
    log export is newer and not yet universally supported, particularly in older
    or managed infrastructure.

  Separation of concerns: log shipping is an infrastructure concern, not an
    application concern.  Keeping it in an agent layer means the log backend
    (Loki → Elasticsearch → CloudWatch) can change without touching code.

WHY GRAFANA ALLOY over Fluentd / Fluentbit / Logstash:
  Alloy is Grafana's own collector agent — the successor to Promtail, Grafana
  Agent, and Grafana Agent Flow, unified into a single binary.  It is a
  first-class citizen in the LGTM stack: it speaks native Loki push format,
  has built-in OTEL receiver support, and its River HCL config language is
  purpose-built for pipeline composition.

  Fluentd and Fluentbit are excellent general-purpose shippers but are
  vendor-neutral — Loki integration requires community plugins and non-trivial
  config.  Logstash is JVM-based and optimised for Elasticsearch.  For a
  Grafana stack on a homelab, Alloy is the right tool: maintained by the same
  team as Loki and Grafana, lighter than Logstash, and simpler to configure
  for LGTM than Fluentbit.

--------------------------------------------------------------------------------
WHY route stdlib logging THROUGH structlog?
--------------------------------------------------------------------------------
The existing codebase uses `logging.getLogger(__name__)` everywhere.  Rather
than rewriting every call site, we hook into the stdlib logging machinery via
`ProcessorFormatter`.  This means every existing log call is automatically
structured — no changes needed at call sites.
"""
from __future__ import annotations

import logging
from typing import Any

import structlog
from opentelemetry import trace as _otel_trace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Guard against setup_observability being called more than once per process.
_observability_configured: bool = False


def setup_observability(service_name: str | None = None) -> None:
    """Configure structured logging and (optionally) the OTEL SDK.

    Call once per process, as early as possible — ideally before any other
    imports that create module-level loggers.

    Parameters
    ----------
    service_name:
        Human-readable name shown in traces and log JSON.  Falls back to
        settings.otel_service_name (which reads OTEL_SERVICE_NAME env var).
    """
    global _observability_configured
    if _observability_configured:
        return
    _observability_configured = True

    from event_driven_rag_service.config.settings import settings

    name = service_name or settings.otel_service_name
    _configure_logging(otel_enabled=settings.otel_enabled)

    if settings.otel_enabled:
        _configure_otel(
            service_name=name,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        )


# ---------------------------------------------------------------------------
# Structlog processor: OTEL trace context injection
# ---------------------------------------------------------------------------

def _inject_otel_context(
    logger: Any,  # noqa: ANN401 — structlog typing convention
    method: str,
    event_dict: dict,
) -> dict:
    """Structlog processor: add trace_id + span_id from the active OTEL span.

    This processor is always registered, even when OTEL is disabled.  When no
    span is active, `span.get_span_context().is_valid` is False, so it's a
    pure no-op.  Inside a span, every log record automatically inherits the
    trace context — no changes needed at log call sites.

    Format: W3C TraceContext hex strings
      trace_id → 32 hex chars (128-bit)
      span_id  → 16 hex chars (64-bit)
    These formats match what Tempo, Loki, and most log aggregators expect.
    """
    span = _otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


# ---------------------------------------------------------------------------
# Internal: structlog + stdlib logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(otel_enabled: bool) -> None:
    """Set up structlog and route stdlib logging through it.

    LEARNING NOTE — two processor chains:
    --------------------------------------
    structlog has two call paths that need separate (but consistent) config:

    1. Native structlog:   log = structlog.get_logger(); log.info("event", k=v)
       → processed by structlog.configure(processors=[...])
       → written directly to stdout via PrintLoggerFactory

    2. Stdlib logging:     logging.getLogger(__name__).info("msg %s", x)
       → intercepted by ProcessorFormatter, processed by foreign_pre_chain + processors
       → written to stdout via StreamHandler

    We define the same shared_processors for both paths so JSON output is
    identical regardless of which API the caller uses.  Grafana Alloy reads both
    from the same stdout stream and pushes them to Loki.
    """
    # Processors that every log record passes through (both call paths)
    shared_processors: list = [
        # merge_contextvars: copies fields bound via structlog.contextvars.bind_contextvars()
        # into every log event — used to propagate trace_id at span start.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,          # "level": "info"
        structlog.stdlib.add_logger_name,        # "logger": "event_driven_rag_service.api.app"
        structlog.processors.TimeStamper(fmt="iso"),  # "timestamp": "2026-05-06T12:34:56.789Z"
        _inject_otel_context,                    # "trace_id" + "span_id" when inside a span
    ]

    if otel_enabled:
        # JSON renderer — what Alloy parses and ships to Loki.
        # Each log line is a complete JSON object on a single line (NDJSON/JSONLines format).
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        # Human-readable, coloured output for `docker compose logs -f` in dev.
        renderer = structlog.dev.ConsoleRenderer()

    # --- Native structlog path ---
    structlog.configure(
        processors=shared_processors + [
            # Handles positional log args: log.info("x=%s", value) → event_dict["event"] = "x=value"
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),    # stack_info= kwarg support
            structlog.processors.ExceptionRenderer(),    # exc_info= kwarg → formatted traceback
            renderer,
        ],
        # make_filtering_bound_logger compiles a bound logger class at import time
        # (no dynamic dispatch per call) — equivalent performance to stdlib logging.
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # --- Stdlib logging path (ProcessorFormatter) ---
    # LEARNING NOTE — ProcessorFormatter anatomy:
    #   foreign_pre_chain: runs ONLY on records from the stdlib logging module
    #   processors:        runs on ALL records (stdlib + native structlog), must end with renderer
    #
    # remove_processors_meta strips structlog's internal bookkeeping fields
    # (_record, _from_structlog, etc.) before the final renderer sees the dict.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Replace ALL existing handlers on the root logger.
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy third-party loggers that produce excessive output at INFO.
    # aio_pika and aiormq log every frame; pika logs every heartbeat.
    # httpx logs every HTTP request (verbose HF model metadata checks).
    logging.getLogger("aio_pika").setLevel(logging.WARNING)
    logging.getLogger("aiormq").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Internal: OTEL SDK configuration
# ---------------------------------------------------------------------------

def _configure_otel(service_name: str, otlp_endpoint: str) -> None:
    """Configure TracerProvider and MeterProvider with OTLP gRPC export.

    Only called when OTEL_ENABLED=true.  All imports are lazy (inside this
    function) so the SDK packages are only needed when actually used.

    Logs are intentionally NOT exported via OTLP — they go to stdout and are
    collected by Grafana Alloy.  See module docstring for the rationale.

    LEARNING NOTE — BatchSpanProcessor vs SimpleSpanProcessor:
    ----------------------------------------------------------
    SimpleSpanProcessor exports each span synchronously as it ends — this
    blocks the thread and adds per-span latency.  Fine for debugging, not
    for production.

    BatchSpanProcessor collects spans in a buffer and exports them on a
    background thread in batches.  If the collector is slow or temporarily
    unreachable, spans queue up in memory (up to max_queue_size) rather than
    blocking your handlers.

    LEARNING NOTE — Resource:
    -------------------------
    A Resource is metadata attached to every span and metric from this
    process.  SERVICE_NAME is the most important attribute — it is what
    Tempo uses to group traces by service.  In a Kubernetes deployment
    you would also add k8s.pod.name, k8s.namespace, deployment.environment, etc.
    via ResourceDetectors.

    LEARNING NOTE — insecure=True:
    ------------------------------
    For homelab/dev we skip TLS (collector is on the same Docker network).
    In production, remove insecure=True and configure mTLS certificates.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    from event_driven_rag_service.config.settings import settings

    resource = Resource.create({SERVICE_NAME: service_name})

    # ── Traces ──────────────────────────────────────────────────────────────
    provider = TracerProvider(resource=resource)
    # OTLP gRPC exporter — sends spans to the OTel Collector (which forwards to Tempo).
    # If the collector is unreachable on startup, the exporter logs a warning and retries
    # in the background — it does NOT crash the service.
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    
    # Select span processor based on OTEL_SPAN_PROCESSOR env var:
    # - "simple": exports spans immediately (low latency, good for debugging)
    # - "batch": buffers spans before export (higher throughput for production)
    if settings.otel_span_processor.lower() == "batch":
        processor = BatchSpanProcessor(exporter)
    else:
        processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # ── Metrics ─────────────────────────────────────────────────────────────
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
    # Export every 15 s — low overhead, fine-grained enough for homelab dashboards.
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    print(f"[observability] OTEL SDK configured: service={service_name}, "
          f"endpoint={otlp_endpoint}, signals=traces+metrics (logs->stdout->Alloy->Loki)")
