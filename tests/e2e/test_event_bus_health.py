"""Health checks for task queue (RabbitMQ) and event bus (PostgresEventBus).

Simple smoke tests to confirm both messaging layers are reachable and functional.
"""
import asyncio
import pytest
import asyncpg

from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# PostgresEventBus (event log)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_bus_setup(postgres_pool_e2e: asyncpg.Pool):
    """Verify event_log and consumer_offsets tables can be created."""
    bus = PostgresEventBus(postgres_pool_e2e)
    await bus.setup_tables()

    async with postgres_pool_e2e.acquire() as conn:
        for table in ("event_log", "consumer_offsets"):
            count = await conn.fetchval(
                "SELECT COUNT(*)::int FROM information_schema.tables WHERE table_name = $1",
                table,
            )
            assert count == 1, f"Table '{table}' should exist after setup_tables()"


@pytest.mark.asyncio
async def test_event_bus_publish_and_consume(postgres_pool_e2e: asyncpg.Pool):
    """Verify publish → subscribe delivers the event payload."""
    bus = PostgresEventBus(postgres_pool_e2e)
    await bus.setup_tables()

    topic = "health.check"
    group = "test_consumer_group"
    event = {"msg": "hello", "seq": 1}

    await bus.publish(topic, event)

    # subscribe() is an infinite async generator — pull one event with a timeout
    received = None
    async def _consume():
        nonlocal received
        async for payload in bus.subscribe(topic, group):
            received = payload
            return

    await asyncio.wait_for(_consume(), timeout=5)

    assert received == event

    # Cleanup: remove test rows so they don't leak into other tests
    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute("DELETE FROM event_log WHERE topic = $1", topic)
        await conn.execute(
            "DELETE FROM consumer_offsets WHERE topic = $1 AND consumer_group = $2",
            topic, group,
        )
