"""PostDispatcher — translates post.synced events into chunk tasks.

Reads from the event log (Redpanda or Postgres mock) and publishes
ChunkTask messages to the RabbitMQ ingestion exchange.

MVP scope
---------
Only body and summary_title chunking are dispatched here.
Inference / categorisation tasks are intentionally excluded until
the inference pipeline is built.
"""
import logging

import aio_pika

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.tasks.registry import TASK_ROUTES

logger = logging.getLogger(__name__)


class PostDispatcher:
    """
    Handles post.synced events from the event log and fans out chunk tasks.

    post.synced:
        - cpu.chunk.post (task_type=body)           — always (or when body changed)
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
        exchange = await channel.get_exchange(route.exchange)

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
        self, exchange: aio_pika.Exchange, event: dict
    ) -> None:
        post_id = event["post_id"]
        post_table = event["post_table"]
        fields_changed: list[str] = event.get("fields_changed", [])
        has_summary: bool = event.get("has_summary", False)

        # An empty fields_changed means "first sync" — dispatch everything.
        # Otherwise only dispatch when the relevant field changed.
        body_changed = not fields_changed or any(
            f in fields_changed for f in ("body_text", "custom_body")
        )
        summary_changed = has_summary and (
            not fields_changed or any(
                f in fields_changed for f in ("summary", "title", "custom_title")
            )
        )

        route = TASK_ROUTES["chunk"]

        if body_changed:
            task = ChunkTask(
                task_type="body",
                post_id=post_id,
                post_table=post_table,
                embed_model=EMBED_CONFIGS["body"].model,
                source_event_id=event.get("event_id"),
                trace_id=event.get("trace_id"),
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
            logger.debug("PostDispatcher: dispatched body chunk task for post %d", post_id)

        if summary_changed:
            task = ChunkTask(
                task_type="summary_title",
                post_id=post_id,
                post_table=post_table,
                embed_model=EMBED_CONFIGS["summary_title"].model,
                source_event_id=event.get("event_id"),
                trace_id=event.get("trace_id"),
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
            logger.debug(
                "PostDispatcher: dispatched summary_title chunk task for post %d", post_id
            )
