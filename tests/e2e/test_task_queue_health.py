"""RabbitMQ health checks for the Docker Compose stack.

Pre-flight smoke tests confirming the task queue is reachable and functional:
connection, queue declaration, and a publish/consume roundtrip.
"""
import json
import pytest
import aio_pika

from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# RabbitMQ task queue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rabbitmq_is_running(rmq_connection_e2e: aio_pika.Connection):
    """Verify RabbitMQ is reachable."""
    assert rmq_connection_e2e.connected, "RabbitMQ connection should be open"


@pytest.mark.asyncio
async def test_can_declare_queue(rmq_connection_e2e: aio_pika.Connection):
    """Verify we can declare exchanges and queues."""
    channel = await rmq_connection_e2e.channel()
    try:
        exchange = await channel.declare_exchange(
            name="test_health_exchange",
            type=aio_pika.ExchangeType.DIRECT,
            durable=False,
            auto_delete=True,
        )
        queue = await channel.declare_queue(
            name="test_health_queue",
            durable=False,
            auto_delete=True,
        )
        await queue.bind(exchange, routing_key="test")
        assert queue.name == "test_health_queue"
    finally:
        await channel.close()


@pytest.mark.asyncio
async def test_can_publish_and_consume(rmq_connection_e2e: aio_pika.Connection):
    """Verify basic publish/consume round-trip."""
    channel = await rmq_connection_e2e.channel()
    try:
        exchange = await channel.declare_exchange(
            name="test_pubsub_exchange",
            type=aio_pika.ExchangeType.DIRECT,
            durable=False,
            auto_delete=True,
        )
        queue = await channel.declare_queue(
            name="test_pubsub_queue",
            durable=False,
            auto_delete=True,
        )
        await queue.bind(exchange, routing_key="test_message")

        test_payload = {"test": "data", "value": 42}
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(test_payload).encode(),
                content_type="application/json",
            ),
            routing_key="test_message",
        )

        # queue.get() fetches a single message without an iterator
        received = await queue.get(timeout=5)
        assert received is not None
        payload = json.loads(received.body.decode())
        assert payload == test_payload
        await received.ack()
    finally:
        await channel.close()
