import aio_pika
from aio_pika import ExchangeType

# ---------------------------------------------------------------------------
# Exchanges
# All work exchanges are direct — routing key == queue name, explicit and
# debuggable. No wildcard routing needed; each dispatcher knows exactly which
# queue it is targeting.
# The dead-letter exchange (dlx) is also direct — rejected/expired messages
# route to their corresponding dlq.* queue by exact key match.
# ---------------------------------------------------------------------------

EXCHANGES: dict[str, str] = {
    "ingestion": str(ExchangeType.DIRECT),  # CPU: post chunking and pre-processing
    "embedding": str(ExchangeType.DIRECT),  # GPU: text embedding, routed by model
    "inference": str(ExchangeType.DIRECT),  # GPU local + IO API inference tasks
    "search":    str(ExchangeType.DIRECT),  # CPU: search execution and ranking
    "dlx":       str(ExchangeType.DIRECT),  # Dead-letter exchange (catches rejected/expired messages)
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
        ("gpu.embed.bge-base-v1.5",         "gpu.embed.bge-base-v1.5"),
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


async def setup_topology(channel: aio_pika.Channel) -> None:
    """Declare all exchanges, work queues, DLQs, and bindings in dependency order."""

    # 1. Declare all exchanges (dlx must exist before work queues reference it)
    declared: dict[str, aio_pika.Exchange] = {}
    for name, kind in EXCHANGES.items():
        declared[name] = await channel.declare_exchange(name, kind, durable=True)

    # 2. Declare DLQs and bind to dlx
    for dlq_name in DLQ_QUEUES:
        dlq = await channel.declare_queue(dlq_name, durable=True)
        await dlq.bind(declared["dlx"], routing_key=dlq_name)

    # 3. Declare work queues (with dlx args) and bind to their exchange
    for exchange_name, bindings in BINDINGS.items():
        exchange = declared[exchange_name]
        for routing_key, queue_name in bindings:
            queue = await channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange":    "dlx",
                    "x-dead-letter-routing-key": f"dlq.{queue_name}",
                },
            )
            await queue.bind(exchange, routing_key=routing_key)