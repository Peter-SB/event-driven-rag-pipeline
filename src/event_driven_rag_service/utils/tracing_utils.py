"""
Distributed tracing helpers — Phase 2: Trace context propagation.

An important problem in distributed tracing is keeping one
trace_id flowing across every service boundary.  This module handles both sides:

  RECORDING — when your span starts, stamp its trace_id + span_id into the
               event/task dict so the next service can find them.

  RESTORING — when you receive an event/task with a trace_id, reconstruct the
               OTEL parent context so your new span becomes a *child* of the
               previous service's span.

Without both sides, Jaeger shows disconnected spans under the same trace
instead of a proper parent → child waterfall.

--------------------------------------------------------------------------------
THE CORE PROBLEM: Kafka and RabbitMQ do NOT propagate trace context
--------------------------------------------------------------------------------
HTTP has the W3C ``traceparent`` header; async transports have nothing.
You must put the trace context into your *data model* and extract it yourself.
This is not a workaround — it's the correct design for event-driven systems.

Auto-instrumentation libraries CAN inject/extract W3C headers into Kafka
message headers, but they don't know about your application-level schema.
Manual propagation gives you:
  - Full control over what gets traced
  - No dependency on transport-level features
  - Visible, debuggable context in every event and task payload
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from opentelemetry import trace, context as otel_context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, INVALID_SPAN_ID


# ---------------------------------------------------------------------------
# Root span decorator
# ---------------------------------------------------------------------------

def traced(span_name: str) -> Callable:
    """Decorator: wrap an async function in an OTEL span.

    Designed for ROOT spans (API endpoints) where there is no incoming parent
    context.  For CONTINUATIONS (dispatchers, handlers receiving events/tasks),
    use ``extract_trace_context()`` + ``tracer.start_as_current_span(name, context=ctx)``
    directly — the parent context must be injected explicitly.

    When OTEL is disabled (OTEL_ENABLED=false), the OTEL API returns a
    NonRecordingSpan.  The decorator still executes but produces zero telemetry
    and adds ~1μs overhead per call — negligible.

    Usage:
        @traced("sync_posts")
        async def sync_posts(req, request):
            # The span is now active.  Read it via trace.get_current_span()
            # or use current_trace_ids() to stamp context onto outgoing events.
            trace_id, span_id = current_trace_ids()
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # get_tracer(__name__) returns the tracer registered by setup_observability().
            # When OTEL is disabled, this is a no-op tracer — same call, zero cost.
            tracer = trace.get_tracer(fn.__module__)
            with tracer.start_as_current_span(span_name):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context extraction — the "restore" side
# ---------------------------------------------------------------------------

def extract_trace_context(
    trace_id_hex: str | None,
    parent_span_id_hex: str | None = None,
) -> otel_context.Context | None:
    """Reconstruct an OTEL parent context from a hex trace_id and optional span_id.

    Returns None when trace_id is missing or unparseable — callers treat None
    as "no parent" (creates a new root span under a fresh trace).

    Usage in a dispatcher:
        ctx = extract_trace_context(event.get("trace_id"), event.get("parent_span_id"))
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("dispatch_chunk_tasks", context=ctx) as span:
            ...

    LEARNING NOTE — why you need parent_span_id, not just trace_id:
    ---------------------------------------------------------------
    A trace_id alone groups spans under the same trace, but does not express
    the parent → child relationship.  Without parent_span_id, Jaeger renders
    all spans as siblings at the same level — no waterfall.

    parent_span_id is the span_id of the PREVIOUS service's span (the one that
    published the event).  Setting it here causes the OTEL SDK to set our new
    span's ``parentSpanId`` to that value, creating the arrow in the Jaeger UI.

    When only trace_id is available (legacy events), we use INVALID_SPAN_ID (0)
    as fallback.  Jaeger still groups these under the same trace, just without
    the parent arrow.

    LEARNING NOTE — is_remote=True:
    --------------------------------
    This flag tells the SDK that the parent span ran in a DIFFERENT process.
    Some backends use this to compute cross-service latency correctly.  Always
    set it to True when restoring context from an event or task.

    LEARNING NOTE — NonRecordingSpan:
    ----------------------------------
    We wrap the reconstructed SpanContext in a NonRecordingSpan before passing
    it as the parent.  This is a sentinel object that carries the context IDs
    without itself being a real span — it will never export telemetry.  The
    SDK uses it only to set the parentSpanId on the real span we're about to
    create.
    """
    if not trace_id_hex:
        return None

    try:
        trace_id = int(trace_id_hex, 16)
    except (ValueError, TypeError):
        return None

    if parent_span_id_hex:
        try:
            span_id = int(parent_span_id_hex, 16)
        except (ValueError, TypeError):
            span_id = INVALID_SPAN_ID
    else:
        span_id = INVALID_SPAN_ID

    span_ctx = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    if not span_ctx.is_valid:
        return None

    return trace.set_span_in_context(NonRecordingSpan(span_ctx))


# ---------------------------------------------------------------------------
# Context injection — the "record" side
# ---------------------------------------------------------------------------

def propagate_trace(fallback_trace_id: str | None) -> tuple[str | None, str | None]:
    """Return the current span's trace ids, falling back to a known trace_id.

    Use this at event/task publication boundaries:
      - When a span is active (OTEL enabled): returns the span's trace_id + span_id
      - When no span is active (OTEL disabled): preserves the incoming trace_id
        so it is not silently dropped during propagation

    This ensures trace_id flows through the pipeline even when OTEL_ENABLED=false,
    which is important for manual correlation via log search.

    Usage:
        trace_id, parent_span_id = propagate_trace(task.trace_id)
        event = ChunksCreatedEvent(..., trace_id=trace_id, parent_span_id=parent_span_id)
    """
    trace_id, span_id = current_trace_ids()
    if trace_id is not None:
        return trace_id, span_id
    # No active span — preserve the incoming trace_id without a parent_span_id.
    # parent_span_id=None means downstream will start a new span under the same
    # trace (no parent arrow in Jaeger), which is acceptable when OTEL is off.
    return fallback_trace_id, None


def current_trace_ids() -> tuple[str | None, str | None]:
    """Return (trace_id_hex, span_id_hex) from the currently active OTEL span.

    Returns (None, None) when no span is active (OTEL disabled, or called
    outside a ``with tracer.start_as_current_span(...)`` block).

    Usage when building outgoing events or tasks:
        trace_id, parent_span_id = current_trace_ids()
        event = PostSyncedEvent(
            ...,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )

    LEARNING NOTE — the "ambient context" pattern:
    -----------------------------------------------
    You do NOT pass the span object around as a function argument.  Instead,
    the span is stored in the coroutine's execution context (Python contextvars)
    by the ``with tracer.start_as_current_span(...)`` block.  Any code called
    inside that block — including this function — can read it with
    ``trace.get_current_span()``.

    This is what "ambient context" means: the span is available everywhere
    in the call stack without being explicitly threaded through every argument.
    In async Python, each coroutine gets its own copy of the context, so spans
    don't leak between concurrent requests.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
