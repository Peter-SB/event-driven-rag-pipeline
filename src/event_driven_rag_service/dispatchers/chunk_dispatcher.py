"""ChunkDispatcher — translates chunks.created events into embed tasks.

Reads from the event log and publishes EmbedTask messages to the RabbitMQ
embedding exchange, routed to the model-specific GPU queue.

Single responsibility: translate chunk-ready events into embedding tasks.
No work is done here.
"""
import logging

import aio_pika
from opentelemetry import trace

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.tasks.registry import TASK_ROUTES
from event_driven_rag_service.utils.tracing_utils import extract_trace_context, propagate_trace

logger = logging.getLogger(__name__)


class ChunkDispatcher:
    """
    Dispatches embedding tasks whenever chunks are ready.

    chunks.created: fired by CpuChunkWorker after a batch of chunks is stored.
    Routes to the model-specific GPU embedding queue via TASK_ROUTES.
    """

    def __init__(self, rmq_connection: aio_pika.Connection, event_bus: EventBusBase) -> None:
        self._event_bus = event_bus
        self._rmq = rmq_connection

    async def run(self) -> None:
        await self._handle_chunks_created()

    async def _handle_chunks_created(self) -> None:
        channel = await self._rmq.channel()
        route = TASK_ROUTES["embed"]
        exchange = await channel.declare_exchange(route.exchange, aio_pika.ExchangeType.TOPIC, durable=True)

        async for event in self._event_bus.subscribe(
            "chunks.created", consumer_group=consumer_groups.CHUNKS_CREATED
        ):
            try:
                await self._dispatch_embedding(exchange, route, event)
            except Exception:
                logger.exception(
                    "ChunkDispatcher: failed to dispatch embed task for post_id=%s",
                    event.get("post_id"),
                )

    async def _dispatch_embedding(
        self,
        exchange: aio_pika.abc.AbstractExchange,
        route,
        event: dict,
    ) -> None:
        parent_ctx = extract_trace_context(
            event.get("trace_id"), event.get("parent_span_id")
        )
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("chunk_dispatcher.dispatch", context=parent_ctx) as span:
            task_type = event.get("task_type", "body")
            embed_cfg = EMBED_CONFIGS.get(task_type, EMBED_CONFIGS["body"])

            span.set_attribute("post_id", event["post_id"])
            span.set_attribute("chunk_count", len(event["chunk_ids"]))
            span.set_attribute("model", embed_cfg.model)

            trace_id, parent_span_id = propagate_trace(event.get("trace_id"))

            task = EmbedTask(
                task_type="chunk",
                model_name=embed_cfg.model,
                post_id=event["post_id"],
                post_table=event["post_table"],
                chunk_ids=event["chunk_ids"],
                chunk_table=event["chunk_table"],
                source_event_id=event.get("event_id"),
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )

            routing_key = route.resolve_key(task)
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=routing_key,
            )
            logger.debug(
                "ChunkDispatcher: dispatched embed task (%d chunks → %s)",
                len(event["chunk_ids"]),
                routing_key,
            )

