"""Tests for distributed trace context propagation utilities.

Verified behaviours
-------------------
- current_trace_ids() returns None when no span is active
- current_trace_ids() returns hex strings inside an active span
- extract_trace_context() returns None for missing/invalid input
- extract_trace_context() reconstructs a valid parent context
- A span created with a restored context has the correct parent trace_id + span_id
  (the "waterfall" relationship that Jaeger renders as parent → child)
- The @traced decorator creates a named span and passes through return values

LEARNING NOTE — InMemorySpanExporter:
---------------------------------------
This exporter is provided by the OTEL SDK for testing.  Instead of sending spans
over the network to a collector, it stores them in a list.  After running the code
under test, you inspect ``exporter.get_finished_spans()`` to assert on span names,
attributes, and parent relationships.

Use SimpleSpanProcessor (not BatchSpanProcessor) in tests because:
  - Simple: synchronous export, span is immediately available after the `with` block closes
  - Batch: asynchronous background thread, span may not be exported by the time you assert

This is one of the few places where SimpleSpanProcessor is the right choice.
"""
from __future__ import annotations

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from event_driven_rag_service.utils.tracing_utils import (
    current_trace_ids,
    extract_trace_context,
    traced,
)


# ---------------------------------------------------------------------------
# Fixture: isolated OTEL provider with in-memory export
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _otel_session_provider():
    """Register an InMemorySpanExporter as the global OTEL provider once per session.

    LEARNING NOTE — why session scope, not function scope:
    -------------------------------------------------------
    The OTEL API allows ``set_tracer_provider()`` exactly ONCE per process:
    it replaces the default ``ProxyTracerProvider`` with a real SDK provider.
    Attempting to replace it a second time (e.g., in teardown to "restore")
    logs a warning and returns without effect.

    The safe pattern for testing is:
      1. Set the real provider once at session start.
      2. Clear the exporter between tests (not the provider).
      3. Never try to restore the original provider.

    In production this is fine — there's only one provider per process lifetime,
    set once during startup by ``setup_observability()``.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def otel_exporter(_otel_session_provider: InMemorySpanExporter):
    """Yield a clean exporter: clears accumulated spans before each test."""
    _otel_session_provider.clear()
    yield _otel_session_provider


def _get_tracer():
    return otel_trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# current_trace_ids
# ---------------------------------------------------------------------------

def test_current_trace_ids_returns_none_outside_span():
    # When no span is active, OTEL returns a NonRecordingSpan whose context
    # is invalid — current_trace_ids() should surface that as (None, None).
    trace_id, span_id = current_trace_ids()
    assert trace_id is None
    assert span_id is None


def test_current_trace_ids_returns_hex_strings_inside_span(otel_exporter):
    tracer = _get_tracer()
    with tracer.start_as_current_span("test_span"):
        trace_id, span_id = current_trace_ids()

    assert trace_id is not None
    assert span_id is not None
    # W3C TraceContext: trace_id = 32 hex chars, span_id = 16 hex chars
    assert len(trace_id) == 32
    assert len(span_id) == 16
    # Both must be valid hex
    int(trace_id, 16)
    int(span_id, 16)


# ---------------------------------------------------------------------------
# extract_trace_context
# ---------------------------------------------------------------------------

def test_extract_trace_context_returns_none_for_none_trace_id():
    assert extract_trace_context(None) is None
    assert extract_trace_context(None, "abcd1234abcd1234") is None


def test_extract_trace_context_returns_none_for_invalid_hex():
    assert extract_trace_context("not-hex") is None
    assert extract_trace_context("zzzz") is None


def test_extract_trace_context_requires_both_ids_for_valid_context():
    # OTEL SDK CONSTRAINT: SpanContext.is_valid requires BOTH trace_id AND span_id
    # to be non-zero.  With trace_id alone (span_id=0 / INVALID_SPAN_ID), the
    # SpanContext is invalid and extract_trace_context returns None.
    #
    # Implication: trace_id-only propagation does NOT create linked spans in Jaeger.
    # It only provides log correlation (via the trace_id field in log records).
    # To get a proper parent→child waterfall, both trace_id AND parent_span_id
    # must be present — which is why both fields exist on BaseEvent/BaseTask.
    trace_id_hex = "a" * 32
    assert extract_trace_context(trace_id_hex) is None        # trace_id alone → None
    assert extract_trace_context(trace_id_hex, "0" * 16) is None  # all-zero span_id → None

    # With a real (non-zero) span_id, a valid context is produced.
    ctx = extract_trace_context(trace_id_hex, "b" * 16)
    assert ctx is not None


# ---------------------------------------------------------------------------
# Parent → child span relationship (the Jaeger waterfall)
# ---------------------------------------------------------------------------

def test_child_span_inherits_trace_id_from_parent(otel_exporter):
    """The core Phase 2 invariant: child spans share the parent's trace_id.

    Flow being tested:
        Service A: creates root span → stamps trace_id + span_id onto event
        Service B: calls extract_trace_context(trace_id, span_id) → creates child span

    In Jaeger, the child span appears indented under the parent with a connecting line.
    """
    tracer = _get_tracer()

    # --- Service A: create root span and capture its context ---
    with tracer.start_as_current_span("service_a_span"):
        trace_id, parent_span_id = current_trace_ids()

    # --- Service B: restore context and create a child span ---
    parent_ctx = extract_trace_context(trace_id, parent_span_id)
    with tracer.start_as_current_span("service_b_span", context=parent_ctx):
        pass

    spans = otel_exporter.get_finished_spans()
    span_a = next(s for s in spans if s.name == "service_a_span")
    span_b = next(s for s in spans if s.name == "service_b_span")

    # Both spans are under the same trace.
    assert span_b.context.trace_id == span_a.context.trace_id

    # span_b's parent is span_a — this is the arrow in Jaeger's waterfall view.
    assert span_b.parent is not None
    assert span_b.parent.span_id == span_a.context.span_id


def test_missing_parent_span_id_creates_independent_trace(otel_exporter):
    """Without parent_span_id, extract_trace_context returns None — spans cannot be linked.

    OTEL SDK CONSTRAINT: a valid SpanContext requires BOTH a non-zero trace_id AND a
    non-zero span_id.  When only trace_id is available (legacy events or log-only
    correlation), extract_trace_context() returns None, and a span created with
    context=None starts a brand-new root trace with a different trace_id.

    Implication for this system:
      - Pre-Phase-2 events (no parent_span_id) cannot appear in the Jaeger waterfall.
      - The trace_id value stored on the event/task still provides LOG correlation:
        you can grep logs by trace_id even when spans are not linked.
      - For a proper parent→child arrow in Jaeger, BOTH trace_id AND parent_span_id
        must be propagated — which is why BaseEvent and BaseTask carry both fields.
    """
    tracer = _get_tracer()

    with tracer.start_as_current_span("origin_span"):
        trace_id, _ = current_trace_ids()  # Discard span_id intentionally

    # trace_id alone → extract_trace_context returns None (span_id=0 → is_valid=False)
    parent_ctx = extract_trace_context(trace_id)  # No parent_span_id
    assert parent_ctx is None  # Cannot reconstruct a valid parent context without span_id

    # With context=None, OTEL starts a completely new root span — new trace_id.
    with tracer.start_as_current_span("continuation_span", context=parent_ctx):
        pass

    spans = otel_exporter.get_finished_spans()
    origin = next(s for s in spans if s.name == "origin_span")
    continuation = next(s for s in spans if s.name == "continuation_span")

    # Different trace_ids: OTEL cannot link spans without a valid parent_span_id.
    # Log correlation is still possible by matching the trace_id string in log records.
    assert continuation.context.trace_id != origin.context.trace_id
    assert continuation.parent is None


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_traced_decorator_creates_named_span(otel_exporter):
    """@traced("name") wraps the function in a span of that name."""

    @traced("my_operation")
    async def do_work() -> str:
        return "done"

    result = await do_work()

    assert result == "done"
    spans = otel_exporter.get_finished_spans()
    assert any(s.name == "my_operation" for s in spans)


@pytest.mark.asyncio
async def test_traced_decorator_span_is_active_during_function(otel_exporter):
    """Inside a @traced function, current_trace_ids() returns valid IDs."""

    captured: list[tuple] = []

    @traced("active_span_test")
    async def capture_ids():
        captured.append(current_trace_ids())

    await capture_ids()

    trace_id, span_id = captured[0]
    assert trace_id is not None
    assert span_id is not None


@pytest.mark.asyncio
async def test_traced_decorator_propagates_exceptions(otel_exporter):
    """@traced does not suppress exceptions from the wrapped function."""

    @traced("failing_op")
    async def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        await fail()

    # The span is still recorded even when the function raises.
    spans = otel_exporter.get_finished_spans()
    assert any(s.name == "failing_op" for s in spans)
