# dispatchers/search_dispatcher.py
import logging
from datetime import datetime, UTC

import aio_pika

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import set_queue_lag
from event_driven_rag_service.tasks.embed_task import EmbedTask

logger = logging.getLogger(__name__)


def _record_queue_lag(event: dict, queue_name: str) -> None:
    """Record how long this event sat in the event log before being dispatched."""
    raw = event.get("occurred_at")
    if raw:
        try:
            occurred = datetime.fromisoformat(raw)
            lag = (datetime.now(UTC) - occurred).total_seconds()
            set_queue_lag(max(lag, 0.0), queue_name)
        except (ValueError, TypeError):
            pass


_QUERY_EMBED_CONFIG = EMBED_CONFIGS["query"]


class SearchDispatcher:
    """
    Handles both steps of the search pipeline.

    Step 1: search_job.created → gpu.embed.{model}  (embed the query)
    Step 2: search_query.embedded → cpu.search.run  (execute search with the vector)

    Single responsibility: translate search events into tasks. No work is done here.
    """

    def __init__(self, rmq_connection: aio_pika.Connection, event_bus: EventBusBase) -> None:
        self._event_bus = event_bus
        self._rmq = rmq_connection

    async def run(self) -> None:
        channel = await self._rmq.channel()
        embedding_ex = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)

        async for event in self._event_bus.subscribe(
            "search_job.created", consumer_group=consumer_groups.SEARCH_JOB_CREATED
        ):
            _record_queue_lag(event, "search")
            task = EmbedTask(
                task_type="query",
                model_name=_QUERY_EMBED_CONFIG.model,
                query=event["query"],
                query_job_id=event["query_job_id"],
                trace_id=event.get("trace_id"),
            )
            await embedding_ex.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=_QUERY_EMBED_CONFIG.queue,
            )
            logger.debug(
                "SearchDispatcher: dispatched query embed task for job_id=%s",
                event.get("query_job_id"),
            )
