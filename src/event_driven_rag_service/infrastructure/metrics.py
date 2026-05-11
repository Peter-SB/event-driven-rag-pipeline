"""Golden metrics for the RAG pipeline.

This module centralizes metric definitions and follows senior observability principles:
1. Only metrics that answer real operational questions are defined
2. Labels are low-cardinality (no user IDs, post IDs, etc.)
3. Ownership is clear (who owns what metric)
4. Histograms are used for latency (not just counters)

CRITICAL: Metrics are created lazily (on first access) so they bind to the real
MeterProvider configured by setup_observability(), not the default no-op meter.
"""
from __future__ import annotations

from typing import Any

from opentelemetry import metrics

# Lazy metric cache — created on first access, not at import time
_metrics_cache: dict[str, Any] = {}


def _get_or_create_metric(name: str, metric_type: str, **kwargs) -> Any:
    """Lazily create and cache a metric instrument.

    This ensures metrics are created AFTER setup_observability() configures the
    real MeterProvider, not at import time when the no-op meter is active.
    """
    if name not in _metrics_cache:
        meter = metrics.get_meter("rag-pipeline", version="1.0.0")
        if metric_type == "counter":
            _metrics_cache[name] = meter.create_counter(name=name, **kwargs)
        elif metric_type == "histogram":
            _metrics_cache[name] = meter.create_histogram(name=name, **kwargs)
        elif metric_type == "gauge":
            _metrics_cache[name] = meter.create_gauge(name=name, **kwargs)
    return _metrics_cache[name]

# ---------------------------------------------------------------------------
# Golden Metrics — the minimum viable observability
# ---------------------------------------------------------------------------
# Rule: Every metric must answer ONE of these questions:
# 1. Is the pipeline healthy? (success/failure rates)
# 2. Where is the bottleneck? (latency histograms)
# 3. Is the system backlogged? (queue depth)
# 4. Is RAG quality degrading? (retrieval hits, reuse, errors by type)

def _posts_processed_total():
    return _get_or_create_metric(
        "rag_posts_processed_total",
        "counter",
        description="Total posts received by the API",
        unit="1",
    )


def _chunks_created_total():
    return _get_or_create_metric(
        "rag_chunks_created_total",
        "counter",
        description="Total chunks created and stored",
        unit="1",
    )


def _embeddings_generated_total():
    return _get_or_create_metric(
        "rag_embeddings_generated_total",
        "counter",
        description="Total embeddings computed and persisted",
        unit="1",
    )


def _failures_total():
    return _get_or_create_metric(
        "rag_failures_total",
        "counter",
        description="Total failures across all pipeline stages",
        unit="1",
    )


def _pipeline_latency_seconds():
    return _get_or_create_metric(
        "rag_pipeline_latency_seconds",
        "histogram",
        description="End-to-end latency from API request to event emission",
        unit="s",
    )


def _chunking_latency_seconds():
    return _get_or_create_metric(
        "rag_chunking_latency_seconds",
        "histogram",
        description="Time to fetch, chunk, deduplicate, and persist",
        unit="s",
    )


def _embedding_latency_seconds():
    return _get_or_create_metric(
        "rag_embedding_latency_seconds",
        "histogram",
        description="Time to encode texts and store vectors",
        unit="s",
    )


def _queue_lag_seconds():
    return _get_or_create_metric(
        "rag_queue_lag_seconds",
        "gauge",
        description="Wall-clock lag from event creation to worker processing",
        unit="s",
    )


def _chunks_deduplicated_total():
    return _get_or_create_metric(
        "rag_chunks_deduplicated_total",
        "counter",
        description="Chunks skipped because text_hash unchanged (idempotency)",
        unit="1",
    )


def _search_jobs_total():
    return _get_or_create_metric(
        "rag_search_jobs_total",
        "counter",
        description="Total search jobs by lifecycle status",
        unit="1",
    )


def _search_latency_seconds():
    return _get_or_create_metric(
        "rag_search_latency_seconds",
        "histogram",
        description="Time from search job creation to results stored",
        unit="s",
    )


def _dlq_messages_total():
    return _get_or_create_metric(
        "rag_dlq_messages_total",
        "counter",
        description="Total messages routed to the dead-letter queue after processing failure",
        unit="1",
    )


def _embedding_encode_seconds():
    return _get_or_create_metric(
        "rag_embedding_encode_seconds",
        "histogram",
        description="Time spent in the model encoder (GPU/CPU compute only, excludes DB I/O)",
        unit="s",
    )


def _model_load_seconds():
    return _get_or_create_metric(
        "rag_model_load_seconds",
        "histogram",
        description="Time to load an embedding model into memory (GPU or CPU)",
        unit="s",
    )


def _model_unload_seconds():
    return _get_or_create_metric(
        "rag_model_unload_seconds",
        "histogram",
        description="Time to unload an embedding model from memory (garbage collection + cleanup)",
        unit="s",
    )


def _embedding_batch_size():
    return _get_or_create_metric(
        "rag_embedding_batch_size",
        "histogram",
        description="Number of texts in each encoder batch (GPU utilisation indicator)",
        unit="1",
    )


def _embedding_seconds_per_text():
    return _get_or_create_metric(
        "rag_embedding_seconds_per_text",
        "histogram",
        description="Encode latency per individual text (encode_seconds / batch_size)",
        unit="s",
    )


def record_posts_processed(count: int, status: str = "success") -> None:
    """Record posts received by the API.

    Status: 'success' | 'error' (validation/db failures before processing)
    """
    _posts_processed_total().add(count, {"status": status})


def record_chunks_created(count: int, task_type: str = "body") -> None:
    """Record new chunks created by a ChunkTask.

    task_type: 'body' | 'title' | 'summary_title' | 'analysis'
    """
    _chunks_created_total().add(count, {"task_type": task_type})


def record_chunks_deduplicated(count: int, task_type: str = "body") -> None:
    """Record chunks skipped due to unchanged text_hash (idempotency metric)."""
    _chunks_deduplicated_total().add(count, {"task_type": task_type})


def record_embeddings_generated(count: int, model: str = "bge-base-v1.5") -> None:
    """Record embeddings computed and stored.

    model: the embedding model name (no version in label for lower cardinality)
    """
    _embeddings_generated_total().add(count, {"model": model})


def record_failure(failure_type: str, service: str = "unknown") -> None:
    """Record a failure.

    failure_type: 'fetch_failed' | 'encode_failed' | 'storage_failed' | 'validation_failed'
    service: 'api' | 'dispatcher' | 'cpu-worker' | 'gpu-worker'
    """
    _failures_total().add(1, {"failure_type": failure_type, "service": service})


def record_pipeline_latency(latency_seconds: float, service: str = "unknown") -> None:
    """Record request-to-event latency."""
    _pipeline_latency_seconds().record(latency_seconds, {"service": service})


def record_chunking_latency(latency_seconds: float, task_type: str = "body") -> None:
    """Record chunking operation latency (fetch + chunk + persist)."""
    _chunking_latency_seconds().record(
        latency_seconds, {"task_type": task_type}
    )


def record_embedding_latency(latency_seconds: float, model: str = "bge-base-v1.5") -> None:
    """Record embedding operation latency (encode + store)."""
    _embedding_latency_seconds().record(latency_seconds, {"model": model})


def set_queue_lag(lag_seconds: float | None, queue_name: str = "chunk") -> None:
    """Set current queue lag (event creation → worker processing).

    This is a gauge (point-in-time measurement), not a cumulative counter.
    lag_seconds: None means queue is empty.
    """
    if lag_seconds is not None:
        _queue_lag_seconds().set(lag_seconds, {"queue": queue_name})


def record_search_job_created() -> None:
    """Record a search job accepted by the API."""
    _search_jobs_total().add(1, {"status": "created"})


def record_search_job_completed(status: str = "completed") -> None:
    """Record a search job terminal outcome.

    status: 'completed' | 'failed'
    """
    _search_jobs_total().add(1, {"status": status})


def record_search_latency(latency_seconds: float) -> None:
    """Record end-to-end search handler latency (job received → results stored)."""
    _search_latency_seconds().record(latency_seconds)


def record_dlq_routed(queue_name: str) -> None:
    """Record a message nacked to the dead-letter queue.

    queue_name: the RabbitMQ queue the message was originally consumed from.
    """
    _dlq_messages_total().add(1, {"queue": queue_name})


def record_encode_latency(latency_seconds: float, model: str, batch_size: int) -> None:
    """Record GPU/CPU encoder time and derived per-text latency.

    Separates compute time from I/O so slow DB writes don't mask fast GPUs.
    batch_size is recorded as a separate metric for GPU utilisation analysis.
    """
    _embedding_encode_seconds().record(latency_seconds, {"model": model})
    _embedding_batch_size().record(batch_size, {"model": model})
    if batch_size > 0:
        _embedding_seconds_per_text().record(latency_seconds / batch_size, {"model": model})


def record_model_load_time(latency_seconds: float, model: str) -> None:
    """Record time to load an embedding model (includes download + initialization)."""
    _model_load_seconds().record(latency_seconds, {"model": model})


def record_model_unload_time(latency_seconds: float, model: str) -> None:
    """Record time to unload an embedding model (garbage collection + cleanup)."""
    _model_unload_seconds().record(latency_seconds, {"model": model})
