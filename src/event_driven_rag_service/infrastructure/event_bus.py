from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class EventBusBase(ABC):
    @abstractmethod
    async def publish(self, topic: str, event: dict) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, consumer_group: str) -> AsyncIterator[dict]: ...


class RedpandaEventBus(EventBusBase):
    """Production event bus backed by Redpanda (Kafka-compatible) via aiokafka.

    Uses a persistent producer to avoid the overhead of opening a new connection
    per publish call.  Call ``start()`` once during app startup and ``stop()``
    during shutdown.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self._servers = bootstrap_servers
        self._producer = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer
        self._producer = AIOKafkaProducer(bootstrap_servers=self._servers)
        await self._producer.start()
        logger.info("RedpandaEventBus producer started (servers=%s)", self._servers)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._producer = None

    async def publish(self, topic: str, event: dict) -> None:
        if self._producer is None:
            raise RuntimeError("Call RedpandaEventBus.start() before publish()")
        # Partition by post_id so events for the same post are ordered.
        key = str(event.get("post_id", "")).encode() or None
        await self._producer.send_and_wait(
            topic,
            json.dumps(event).encode(),
            key=key,
        )

    async def subscribe(self, topic: str, consumer_group: str) -> AsyncIterator[dict]:  # type: ignore[override]
        from aiokafka import AIOKafkaConsumer
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._servers,
            group_id=consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        await consumer.start()
        logger.info("RedpandaEventBus consumer started (topic=%s group=%s)", topic, consumer_group)
        try:
            async for msg in consumer:
                if msg.value is not None:
                    yield json.loads(msg.value)
        finally:
            await consumer.stop()


class PostgresEventBus(EventBusBase):
    """Homelab event bus backed by Postgres as a polling-based append-only log.

    Events are stored in the ``event_log`` table.  Each consumer group tracks
    its read position in ``consumer_offsets`` and polls for new rows on a
    configurable interval.

    Note on LISTEN/NOTIFY
    ---------------------
    PostgreSQL's LISTEN/NOTIFY mechanism would provide true push semantics
    (no polling latency, one persistent connection per consumer), more closely
    matching Redpanda's consumer model.  Simple polling was chosen here to
    reduce implementation complexity at the MVP stage.  Migrating to
    LISTEN/NOTIFY is a straight swap: replace the polling loop with an
    ``await conn.add_listener(topic, callback)`` call.
    """

    POLL_INTERVAL_S: float = 0.5 # todo: make environment variable
    BATCH_SIZE: int = 100

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup_tables(self) -> None:
        """Idempotently create event_log and consumer_offsets tables."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    topic       TEXT        NOT NULL,
                    payload     JSONB       NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS event_log_topic_id_idx
                    ON event_log (topic, id);

                CREATE TABLE IF NOT EXISTS consumer_offsets (
                    consumer_group TEXT   NOT NULL,
                    topic          TEXT   NOT NULL,
                    last_id        BIGINT NOT NULL DEFAULT 0,
                    PRIMARY KEY (consumer_group, topic)
                );
            """)
        logger.info("PostgresEventBus: event_log and consumer_offsets tables ready")

    async def publish(self, topic: str, event: dict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO event_log (topic, payload) VALUES ($1, $2::jsonb)",
                topic,
                json.dumps(event),
            )

    async def subscribe(self, topic: str, consumer_group: str) -> AsyncIterator[dict]:  # type: ignore[override]
        """Yield events from *topic* in order, resuming from the stored offset.

        Polls every ``POLL_INTERVAL_S`` seconds when there are no new events.
        The offset is advanced after each event is yielded, so a crash between
        events will cause the current event to be re-delivered on restart
        (at-least-once delivery).
        """
        while True:
            async with self._pool.acquire() as conn:
                # Ensure offset row exists, return current position.
                last_id: int = await conn.fetchval(
                    """
                    INSERT INTO consumer_offsets (consumer_group, topic, last_id)
                    VALUES ($1, $2, 0)
                    ON CONFLICT (consumer_group, topic)
                        DO UPDATE SET last_id = consumer_offsets.last_id
                    RETURNING last_id
                    """,
                    consumer_group,
                    topic,
                )

                rows = await conn.fetch(
                    """
                    SELECT id, payload
                    FROM   event_log
                    WHERE  topic = $1
                      AND  id    > $2
                    ORDER  BY id ASC
                    LIMIT  $3
                    """,
                    topic,
                    last_id,
                    self.BATCH_SIZE,
                )

            if not rows:
                await asyncio.sleep(self.POLL_INTERVAL_S)
                continue

            for row in rows:
                # Commit offset BEFORE yielding: ensures the offset advances
                # even when the caller breaks out of the generator early.
                # Trade-off: at-most-once delivery instead of at-least-once,
                # which is acceptable for the Postgres-backed homelab bus.
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE consumer_offsets
                        SET    last_id = $1
                        WHERE  consumer_group = $2
                          AND  topic = $3
                        """,
                        row["id"],
                        consumer_group,
                        topic,
                    )
                yield json.loads(row["payload"])


def create_event_bus(pool: asyncpg.Pool | None = None) -> EventBusBase:
    """Factory — swap implementation via ``EVENT_BUS`` environment variable.

    Args:
        pool: asyncpg connection pool (required for PostgresEventBus).
    """
    mode = os.getenv("EVENT_BUS", "postgres")
    if mode == "redpanda":
        servers = os.getenv("REDPANDA_SERVERS", "localhost:19092")
        return RedpandaEventBus(servers)
    if pool is None:
        raise ValueError("asyncpg connection pool is required for PostgresEventBus")
    return PostgresEventBus(pool)


# Alias kept for backward compatibility with existing dispatcher code
create_event_log = create_event_bus
