import logging

import aio_pika
from aio_pika import ExchangeType
from aiormq.exceptions import ChannelPreconditionFailed

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exchanges
# Work exchanges use TOPIC type to support dynamic routing keys (e.g., gpu.embed.{model}).
# Each dispatcher publishes to the appropriate exchange with its routing key pattern.
# The dead-letter exchange (dlx) is direct — rejected/expired messages route to
# their corresponding dlq.* queue by exact key match.
# ---------------------------------------------------------------------------

EXCHANGES: dict[str, str] = {
    "ingestion": ExchangeType.TOPIC,  # CPU: post chunking and pre-processing
    "embedding": ExchangeType.TOPIC,  # GPU: text embedding, routed by model
    "inference": ExchangeType.TOPIC,  # GPU local + IO API inference tasks
    "search":    ExchangeType.TOPIC,  # CPU: search execution and ranking
    "dlx":       ExchangeType.DIRECT,  # Dead-letter exchange (catches rejected/expired messages)
}

# ---------------------------------------------------------------------------
# Bindings
# (routing_key, queue_name) pairs per exchange.
# Routing key mirrors the queue name — publishers use the destination queue
# name as the routing key, which is explicit and self-documenting.
#
# Queue pattern: {worker_type}.{task}.{qualifier}
# DLQ pattern:   dlq.{worker_type}.{task}.{qualifier}
# ---------------------------------------------------------------------------

BINDINGS: dict[str, list[tuple[str, str]]] = {
    "ingestion": [
        # routing_key                       queue_name
        ("cpu.chunk.post",                  "cpu.chunk.post"),
    ],
    "embedding": [
        # One queue per model — workers bind to their model's queue and stay
        # warm. Model swaps only happen when the queue drains.
        ("gpu.embed.bge-base-en-v1.5",       "gpu.embed.bge-base-en-v1.5"),
        ("gpu.embed.bge-small-en-v1.5",     "gpu.embed.bge-small-en-v1.5"),
        ("gpu.embed.qwen3-0.6b",            "gpu.embed.qwen3-0.6b"),
    ],
    "inference": [
        ("gpu.infer_local.qwen3.5-4b",      "gpu.infer_local.qwen3.5-4b"),
        ("io.infer_api.chatgpt-4o",         "io.infer_api.chatgpt-4o"),
    ],
    "search": [
        ("cpu.search.run",                  "cpu.search.run"),
        ("cpu.search.rank",                 "cpu.search.rank"),
    ],
}

# ---------------------------------------------------------------------------
# Dead-letter queues
# Auto-derived from work queues. Bound to dlx with routing key = dlq name.
# Messages that are rejected, expired, or exceed max delivery count land here.
# ---------------------------------------------------------------------------

_WORK_QUEUES: list[str] = [q for bindings in BINDINGS.values() for _, q in bindings]
DLQ_QUEUES:   list[str] = [f"dlq.{q}" for q in _WORK_QUEUES]


def verify_embedding_topology() -> None:
    """Fail loudly if EMBED_CONFIGS and BINDINGS['embedding'] have drifted apart.

    ChunkDispatcher and SearchDispatcher publish embed tasks using
    ``EMBED_CONFIGS[task_type].queue`` as the routing key. If that queue isn't
    declared here, the message is silently dropped by RabbitMQ's topic
    exchange — no exception, no DLQ entry, just missing embeddings. Run this
    at process startup (not just in tests) so a deploy with mismatched config
    crashes immediately instead of failing silently in production.
    """
    declared = {queue for _, queue in BINDINGS["embedding"]}
    configured = {cfg.queue for cfg in EMBED_CONFIGS.values()}

    missing = configured - declared
    if missing:
        raise RuntimeError(
            f"EMBED_CONFIGS references queues not declared in BINDINGS['embedding']: "
            f"{sorted(missing)}. Embed tasks routed to these queues would be silently "
            f"dropped. Add them to task_queue.BINDINGS['embedding']."
        )

    orphaned = declared - configured
    if orphaned:
        logger.warning(
            "BINDINGS['embedding'] declares queues not referenced by any EMBED_CONFIGS "
            "entry (dead queues, safe to remove if intentional): %s",
            sorted(orphaned),
        )


async def setup_topology(connection: aio_pika.abc.AbstractRobustConnection) -> None:
    """Declare all exchanges, work queues, DLQs, and bindings in dependency order.

    Accepts a *connection* (not a channel) so it can open fresh channels when a
    work queue has a stale definition.  RabbitMQ closes the channel with
    PRECONDITION_FAILED if a queue is redeclared with different arguments (e.g.
    a queue that existed before DLX args were added).  Opening a new channel per
    work queue lets us catch that error, delete the stale queue, and redeclare it
    correctly — without aborting the whole topology setup.
    """
    verify_embedding_topology()

    # 1. Declare all exchanges and DLQs on a shared channel.
    #    Neither exchanges nor DLQs carry args that change over time, so
    #    idempotent redeclaration is always safe here.
    async with connection.channel() as channel:
        declared: dict[str, aio_pika.abc.AbstractExchange] = {}
        for name, kind in EXCHANGES.items():
            declared[name] = await channel.declare_exchange(name, kind, durable=True)

        for dlq_name in DLQ_QUEUES:
            dlq = await channel.declare_queue(dlq_name, durable=True)
            await dlq.bind(declared["dlx"], routing_key=dlq_name)

    # 2. Declare work queues with DLX args.  Each queue gets its own channel so
    #    a PRECONDITION_FAILED on one queue doesn't close the shared channel and
    #    abort the rest.
    for exchange_name, bindings in BINDINGS.items():
        for routing_key, queue_name in bindings:
            await _ensure_work_queue(connection, exchange_name, routing_key, queue_name)


async def _ensure_work_queue(
    connection: aio_pika.abc.AbstractRobustConnection,
    exchange_name: str,
    routing_key: str,
    queue_name: str,
) -> None:
    """Declare a work queue with DLX args, deleting stale definitions if needed.

    If the queue already exists with different arguments (e.g. it was originally
    declared without a dead-letter exchange), RabbitMQ raises
    ``ChannelPreconditionFailed`` and closes the channel.  We catch that,
    delete the stale queue on a fresh channel, then redeclare it correctly.
    """
    args = {
        "x-dead-letter-exchange":    "dlx",
        "x-dead-letter-routing-key": f"dlq.{queue_name}",
    }
    try:
        async with connection.channel() as ch:
            queue = await ch.declare_queue(queue_name, durable=True, arguments=args)
            await queue.bind(exchange_name, routing_key=routing_key)
    except ChannelPreconditionFailed:
        logger.warning(
            "Queue %r has stale definition (missing DLX args) — deleting and recreating",
            queue_name,
        )
        async with connection.channel() as ch:
            stale = await ch.declare_queue(queue_name, passive=True)
            await stale.delete(if_unused=False, if_empty=False)
        async with connection.channel() as ch:
            queue = await ch.declare_queue(queue_name, durable=True, arguments=args)
            await queue.bind(exchange_name, routing_key=routing_key)