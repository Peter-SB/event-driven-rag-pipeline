# dispatchers/embedding_dispatcher.py
import logging

import aio_pika

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.tasks.search_tasks import SearchRunTask
from event_driven_rag_service.tasks.registry import TASK_ROUTES

logger = logging.getLogger(__name__)


class EmbeddingDispatcher:
    """
    Listens to search_query.embedded on the event log.

    search_query.embedded → cpu.search.run

    This event is only emitted for query embeddings (never for chunk embeddings),
    so no discriminator check is needed. Chunk embeddings are terminal — no
    downstream task is dispatched from embedding.completed.

    Single responsibility: trigger search execution once a query is embedded.
    """

    def __init__(self, rmq_connection: aio_pika.Connection, event_bus: EventBusBase) -> None:
        self._event_bus = event_bus
        self._rmq = rmq_connection

    async def run(self) -> None:
        channel = await self._rmq.channel()
        search_ex = await channel.declare_exchange("search", aio_pika.ExchangeType.TOPIC, durable=True)

        async for event in self._event_bus.subscribe(
            "search_query.embedded", consumer_group=consumer_groups.SEARCH_QUERY_EMBEDDED
        ):
            route = TASK_ROUTES["search_run"]
            task = SearchRunTask(
                job_id=event["query_job_id"],
                trace_id=event.get("trace_id"),
            )
            await search_ex.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
            logger.debug(
                "EmbeddingDispatcher: dispatched search run task for job_id=%s",
                event.get("query_job_id"),
            )
