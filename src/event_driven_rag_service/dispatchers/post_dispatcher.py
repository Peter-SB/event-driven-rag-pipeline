"""PostDispatcher — translates post.synced events into chunk tasks.

Reads from the event log (Redpanda or Postgres mock) and publishes
ChunkTask messages to the RabbitMQ ingestion exchange.

MVP scope
---------
Body, title, and summary_title chunking are dispatched here.
Inference / categorisation tasks are intentionally excluded until
the inference pipeline is built.
"""
import logging

import aio_pika
from opentelemetry import trace

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.tasks.registry import TASK_ROUTES
from event_driven_rag_service.utils.tracing_utils import extract_trace_context, propagate_trace

logger = logging.getLogger(__name__)


class PostDispatcher:
    """
    Handles post.synced events from the event log and fans out chunk tasks.

    post.synced:
        - cpu.chunk.post (task_type=body)           — always (or when body changed)
        - cpu.chunk.post (task_type=title)          — when title or custom_title changed
        - cpu.chunk.post (task_type=summary_title)  — when has_summary=True

    Single responsibility: translate post events into tasks. No work is done here.
    """

    def __init__(self, rmq_connection: aio_pika.Connection, event_bus: EventBusBase) -> None:
        self._event_bus = event_bus
        self._rmq = rmq_connection

    async def run(self) -> None:
        await self._handle_post_synced()

    async def _handle_post_synced(self) -> None:
        channel = await self._rmq.channel()
        route = TASK_ROUTES["chunk"]
        exchange = await channel.declare_exchange(route.exchange, aio_pika.ExchangeType.TOPIC, durable=True)

        async for event in self._event_bus.subscribe(
            "post.synced", consumer_group=consumer_groups.POST_SYNCED
        ):
            try:
                await self._dispatch_chunk_tasks(exchange, event)
            except Exception:
                logger.exception(
                    "PostDispatcher: failed to dispatch tasks for post_id=%s",
                    event.get("post_id"),
                )

    async def _dispatch_chunk_tasks(
        self, exchange: aio_pika.abc.AbstractExchange, event: dict
    ) -> None:
        # Restore the parent span context from the event so this span becomes
        # a child of the API's sync_posts span.  If the event has no trace_id
        # (e.g., events from before Phase 2), parent_ctx is None and we start
        # a new root span — graceful degradation, no crash.
        parent_ctx = extract_trace_context(
            event.get("trace_id"), event.get("parent_span_id")
        )
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("post_dispatcher.dispatch", context=parent_ctx) as span:
            post_id = event["post_id"]
            post_table = event["post_table"]
            span.set_attribute("post_id", post_id)
            span.set_attribute("post_table", post_table)

            fields_changed: list[str] = event.get("fields_changed", [])
            has_summary: bool = event.get("has_summary", False)

            body_changed = not fields_changed or any(
                f in fields_changed for f in ("body_text", "custom_body")
            )
            title_changed = not fields_changed or any(
                f in fields_changed for f in ("title", "custom_title")
            )
            summary_changed = has_summary and (
                not fields_changed or any(
                    f in fields_changed for f in ("summary", "title", "custom_title")
                )
            )

            # Stamp THIS dispatcher span's context onto each task.
            # Falls back to event["trace_id"] when OTEL is disabled (no active span),
            # so trace_id is never silently dropped during propagation.
            trace_id, parent_span_id = propagate_trace(event.get("trace_id"))

            route = TASK_ROUTES["chunk"]
            await self._route_tasks(
                exchange, event, post_id, post_table,
                body_changed, title_changed, summary_changed,
                route, trace_id, parent_span_id,
            )

    async def _route_tasks(
        self, exchange, event, post_id, post_table,
        body_changed, title_changed, summary_changed,
        route, trace_id: str | None, parent_span_id: str | None,
    ):
        if body_changed:
            task = ChunkTask(
                task_type="body",
                post_id=post_id,
                post_table=post_table,
                embed_model=EMBED_CONFIGS["body"].model,
                source_event_id=event.get("event_id"),
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
            logger.debug("PostDispatcher: dispatched body chunk task for post %d", post_id)

        if title_changed:
            task = ChunkTask(
                task_type="title",
                post_id=post_id,
                post_table=post_table,
                embed_model=EMBED_CONFIGS["title"].model,
                source_event_id=event.get("event_id"),
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
            logger.debug("PostDispatcher: dispatched title chunk task for post %d", post_id)

        if summary_changed:
            task = ChunkTask(
                task_type="summary_title",
                post_id=post_id,
                post_table=post_table,
                embed_model=EMBED_CONFIGS["summary_title"].model,
                source_event_id=event.get("event_id"),
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
            logger.debug(
                "PostDispatcher: dispatched summary_title chunk task for post %d", post_id
            )
