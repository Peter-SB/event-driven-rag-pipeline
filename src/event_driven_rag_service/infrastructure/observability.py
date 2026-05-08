"""
Observability bootstrap for all pipeline services — Phase 1: Foundation.

Implements the "three pillars" (logs, metrics, traces) foundation using
OpenTelemetry + structlog.  Phase 1 focuses on structured logging and SDK
wiring.  Actual span creation and metrics come in Phase 2+.

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
  - structlog renders JSON so log aggregators (Loki, ELK, Datadog) can parse it
  - A real TracerProvider is registered globally with a BatchSpanProcessor
  - Every log record gains trace_id + span_id from the active OTEL span
  - Spans are exported via OTLP gRPC to the configured collector endpoint

--------------------------------------------------------------------------------
WHY structlog OVER plain logging?
--------------------------------------------------------------------------------
Python's stdlib logging produces unstructured text.  Parsing that text in a
log aggregator requires fragile regex.  structlog produces a Python dict first
and renders it last — JSON output is a one-line config change, not a rewrite.
It also integrates cleanly with OTEL by acting as a processor in the chain.

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Guard against setup_observability being called more than once per process.
# This can happen when OTEL SDK imports (grpc, exporter internals) have side
# effects that trigger set_tracer_provider() via the SDK's auto-detection path.
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
    _configure_logging(otel_enabled=settings.otel_enabled, service_name=name)

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

    LEARNING NOTE — why always register this, even when OTEL is disabled:
    -----------------------------------------------------------------------
    This processor is in the chain regardless of OTEL_ENABLED.  When no span
    is active, `span.get_span_context().is_valid` is False, so it's a pure
    no-op.  When Phase 2 adds span creation, every log emitted *inside* a
    span automatically inherits the trace context — developers get correlation
    for free without touching individual log call sites.

    This is the "ambient context" pattern from distributed tracing: context
    flows with the thread/coroutine, not as explicit function arguments.

    Format: W3C TraceContext hex strings
      trace_id → 32 hex chars (128-bit)
      span_id  → 16 hex chars (64-bit)
    These formats match what Jaeger, Tempo, and most log aggregators expect.
    """
    from opentelemetry import trace

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


# ---------------------------------------------------------------------------
# Internal: structlog + stdlib logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(otel_enabled: bool, service_name: str) -> None:
    """Set up structlog and route stdlib logging through it.

    LEARNING NOTE — two processor chains:
    --------------------------------------
    structlog has two call paths that need separate (but consistent) config:

    1. Native structlog:   log = structlog.get_logger(); log.info("event", k=v)
       → processed by structlog.configure(processors=[...])

    2. Stdlib logging:     logging.getLogger(__name__).info("msg %s", x)
       → intercepted by ProcessorFormatter, processed by foreign_pre_chain + processors

    We define the same shared_processors for both paths so JSON output is
    identical regardless of which API the caller uses.
    """
    # Processors that every log record passes through (both call paths)
    shared_processors: list = [
        # merge_contextvars: copies fields bound via structlog.contextvars.bind_contextvars()
        # into every log event.  Used in Phase 2+ to bind trace_id at span start so
        # even log calls that predate the inject processor see it.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,          # "level": "info"
        structlog.stdlib.add_logger_name,        # "logger": "event_driven_rag_service.api.app"
        structlog.processors.TimeStamper(fmt="iso"),  # "timestamp": "2026-05-06T12:34:56.789Z"
        _inject_otel_context,                    # "trace_id" + "span_id" (no-op until Phase 2)
    ]

    if otel_enabled:
        # JSON renderer — what production log aggregators (Loki, CloudWatch, ELK) parse.
        # Each log line is a complete JSON object on a single line (NDJSON/JSONLines format).
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        # Human-readable, coloured output — great for `docker compose logs -f` in dev.
        # ConsoleRenderer adds colour codes and aligns key=value pairs for readability.
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
    # This clears any prior basicConfig() call in the entrypoints — those will be
    # removed from the entrypoints themselves, but this is a safety net.
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy third-party loggers that produce excessive output at INFO.
    # aio_pika and aiormq log every frame; pika logs every heartbeat.
    logging.getLogger("aio_pika").setLevel(logging.WARNING)
    logging.getLogger("aiormq").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Internal: OTEL SDK configuration
# ---------------------------------------------------------------------------

def _configure_otel(service_name: str, otlp_endpoint: str) -> None:
    """Configure a real TracerProvider with a BatchSpanProcessor → OTLP exporter.

    Only called when OTEL_ENABLED=true.  All imports are lazy (inside this
    function) so the SDK packages are only needed when actually used.

    LEARNING NOTE — BatchSpanProcessor vs SimpleSpanProcessor:
    ----------------------------------------------------------
    SimpleSpanProcessor exports each span synchronously as it ends — this
    blocks the thread and adds per-span latency.  Fine for debugging, not
    for production.

    BatchSpanProcessor collects spans in a buffer and exports them on a
    background thread in batches.  This decouples your service's hot path
    from export I/O.  If the collector is slow or temporarily unreachable,
    spans queue up in memory (up to max_queue_size) rather than blocking
    your handlers.

    LEARNING NOTE — Resource:
    -------------------------
    A Resource is metadata attached to every span and metric from this
    process.  SERVICE_NAME is the most important attribute — it's what
    Jaeger uses to group traces by service.  In a Kubernetes deployment
    you'd also add k8s.pod.name, k8s.namespace, deployment.environment, etc.
    via ResourceDetectors.

    LEARNING NOTE — insecure=True:
    ------------------------------
    For homelab/dev we skip TLS (collector is on the same Docker network).
    In production, remove insecure=True and configure mTLS certificates.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource.create({SERVICE_NAME: service_name})

    # ── Traces ──────────────────────────────────────────────────────────────
    provider = TracerProvider(resource=resource)

    # OTLP gRPC exporter — sends spans to the OTel Collector (or directly to Jaeger/Tempo).
    # If the collector is unreachable on startup, the exporter logs a warning and retries
    # in the background — it does NOT crash the service.  This is important for homelab
    # where the collector might not start before the app.
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Register as the global TracerProvider.  After this call, any code anywhere in the
    # process that calls `opentelemetry.trace.get_tracer(__name__)` gets a real tracer
    # connected to this provider.  Before this call, they got a no-op tracer.
    trace.set_tracer_provider(provider)

    # ── Metrics ─────────────────────────────────────────────────────────────
    # Without a real MeterProvider, every instrument in metrics.py is a no-op
    # and nothing is exported to the OTel Collector / Prometheus.
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
    # Export every 15 s — low overhead, fine-grained enough for homelab dashboards.
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Log via print (not structlog) during setup to avoid logger initialization issues
    print(f"[observability] OTEL SDK configured: service={service_name}, "
          f"endpoint={otlp_endpoint}, sampler=always_on")
