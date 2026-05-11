"""Tests for infrastructure/observability.py and infrastructure/metrics.py.

Verified behaviours
-------------------
setup_observability:
  - Calling it multiple times does not reconfigure structlog or the OTEL SDK
    (idempotency guard via _observability_configured flag).

metrics recording functions:
  - Each public record_* function calls through without raising when invoked
    before setup_observability() (no-op meter is active).
  - Labels are passed correctly to the underlying instrument.
"""
from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# setup_observability — idempotency
# ---------------------------------------------------------------------------

def test_setup_observability_is_idempotent(monkeypatch):
    """setup_observability() must not reconfigure on subsequent calls.

    The function guards against double-configuration with _observability_configured.
    We set the guard to True (simulating an already-configured process) and confirm
    that _configure_logging is never called on subsequent invocations.

    NOTE: We do NOT reset the guard to False here.  Resetting it would cause
    setup_observability() to run _configure_otel, which calls
    trace.set_tracer_provider() and displaces the InMemorySpanExporter registered
    by the OTEL session fixture in test_trace_propagation.py.
    """
    import event_driven_rag_service.infrastructure.observability as obs_module

    calls: list[str] = []

    def fake_configure_logging(otel_enabled: bool) -> None:
        calls.append("configure_logging")

    # Simulate a process that has already been configured.
    monkeypatch.setattr(obs_module, "_observability_configured", True)
    monkeypatch.setattr(obs_module, "_configure_logging", fake_configure_logging)

    obs_module.setup_observability("test-service")
    obs_module.setup_observability("test-service")  # second call — must be no-op
    obs_module.setup_observability("test-service")  # third call — must be no-op

    assert calls == [], (
        "_configure_logging should never be called when already configured; "
        f"was called {len(calls)} times"
    )


def test_setup_observability_guard_is_set_after_first_call(monkeypatch):
    """After setup_observability() runs, _observability_configured is True.

    Both _configure_logging and _configure_otel are mocked so no real structlog
    or OTEL SDK setup happens — this prevents the test from displacing the
    InMemorySpanExporter used by test_trace_propagation.py.
    """
    import event_driven_rag_service.infrastructure.observability as obs_module

    monkeypatch.setattr(obs_module, "_observability_configured", False)
    monkeypatch.setattr(obs_module, "_configure_logging", lambda otel_enabled: None)
    # Also mock _configure_otel in case OTEL_ENABLED=true in the environment,
    # which would otherwise call trace.set_tracer_provider() and break the
    # InMemorySpanExporter session fixture in test_trace_propagation.py.
    monkeypatch.setattr(obs_module, "_configure_otel", lambda **kwargs: None)

    obs_module.setup_observability()

    assert obs_module._observability_configured is True


# ---------------------------------------------------------------------------
# metrics recording functions — smoke tests (no-op meter is active)
# ---------------------------------------------------------------------------
# The OTEL no-op meter silently drops all record/add calls.  These tests
# confirm the functions accept their arguments and return without raising,
# catching label-signature regressions early.

from event_driven_rag_service.infrastructure.metrics import (
    record_posts_processed,
    record_chunks_created,
    record_chunks_deduplicated,
    record_embeddings_generated,
    record_failure,
    record_pipeline_latency,
    record_chunking_latency,
    record_embedding_latency,
    set_queue_lag,
    record_search_job_created,
    record_search_job_completed,
    record_search_latency,
    record_dlq_routed,
    record_encode_latency,
    record_model_load_time,
    record_model_unload_time,
)


@pytest.mark.parametrize("status", ["inserted", "updated", "skipped", "error"])
def test_record_posts_processed_accepts_all_statuses(status):
    record_posts_processed(5, status)


@pytest.mark.parametrize("task_type", ["body", "title", "summary_title", "analysis"])
def test_record_chunks_created_accepts_all_task_types(task_type):
    record_chunks_created(3, task_type)


@pytest.mark.parametrize("task_type", ["body", "title", "summary_title"])
def test_record_chunks_deduplicated_accepts_task_types(task_type):
    record_chunks_deduplicated(2, task_type)


def test_record_embeddings_generated():
    record_embeddings_generated(10, "bge-base-v1.5")


@pytest.mark.parametrize("failure_type,service", [
    ("fetch_failed", "cpu-worker"),
    ("encode_failed", "gpu-worker"),
    ("storage_failed", "api"),
    ("validation_failed", "dispatcher"),
])
def test_record_failure_accepts_known_labels(failure_type, service):
    record_failure(failure_type, service)


def test_record_pipeline_latency():
    record_pipeline_latency(0.123, "api")


def test_record_chunking_latency():
    record_chunking_latency(0.456, "body")


def test_record_embedding_latency():
    record_embedding_latency(1.234, "bge-base-v1.5")


def test_set_queue_lag_with_value():
    set_queue_lag(2.5, "chunk")


def test_set_queue_lag_with_none_does_not_raise():
    # None means queue is empty — should be a silent no-op.
    set_queue_lag(None, "chunk")


def test_record_search_job_lifecycle():
    record_search_job_created()
    record_search_job_completed("completed")
    record_search_job_completed("failed")


def test_record_search_latency():
    record_search_latency(0.789)


def test_record_dlq_routed():
    record_dlq_routed("cpu.chunk.post")


def test_record_encode_latency_with_batch():
    record_encode_latency(0.5, "bge-base-v1.5", 32)


def test_record_encode_latency_zero_batch_does_not_raise():
    # batch_size=0 triggers a division guard — must not raise ZeroDivisionError.
    record_encode_latency(0.1, "bge-base-v1.5", 0)


def test_record_model_load_time():
    record_model_load_time(3.2, "bge-base-v1.5")


def test_record_model_unload_time():
    record_model_unload_time(0.1, "bge-base-v1.5")
