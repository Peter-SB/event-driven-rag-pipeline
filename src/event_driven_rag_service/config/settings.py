"""
Application settings loaded from environment variables (or a .env file).

All infrastructure URLs and tuneable parameters live here. Workers, dispatchers,
and the API read from this module — no hardcoded strings in logic code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- Database ---
    db_url: str = field(
        default_factory=lambda: os.getenv(
            "DB_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"
        )
    )
    db_pool_min: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MIN", "2")))
    db_pool_max: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MAX", "10")))

    # --- RabbitMQ ---
    rabbitmq_url: str = field(
        default_factory=lambda: os.getenv(
            "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"
        )
    )

    # --- Event bus ---
    # "postgres" for homelab/dev, "redpanda" for production
    event_bus: str = field(default_factory=lambda: os.getenv("EVENT_BUS", "postgres"))
    redpanda_servers: str = field(
        default_factory=lambda: os.getenv("REDPANDA_SERVERS", "localhost:19092")
    )

    # --- Observability ---
    # Off by default: instrumentation compiles in everywhere, but exporters only
    # start when this is true.  Principle: "instrumentation is cheap, exporting is expensive."
    otel_enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    # Human-readable service name that appears in Jaeger/Tempo trace views and log JSON.
    # Each process overrides this via the OTEL_SERVICE_NAME env var so spans from
    # "rag-api", "rag-dispatcher", and "rag-cpu-worker" are distinct in the trace UI.
    otel_service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "rag-pipeline")
    )
    # OTLP gRPC endpoint for the OpenTelemetry Collector (or a direct Jaeger backend).
    # Port 4317 is the OTLP gRPC standard; 4318 is OTLP HTTP.
    # In docker-compose, this points to the otel-collector service: "http://otel-collector:4317"
    otel_exporter_otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )
    # Span processor: "simple" for fast debugging (exports immediately),
    # or "batch" for production (buffers spans for efficiency).
    otel_span_processor: str = field(
        default_factory=lambda: os.getenv("OTEL_SPAN_PROCESSOR", "simple")
    )


settings = Settings()
