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

    # --- Posts ---
    posts_table: str = field(
        default_factory=lambda: os.getenv("POSTS_TABLE", "posts")
    )


settings = Settings()
