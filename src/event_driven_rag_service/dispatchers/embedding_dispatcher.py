# dispatchers/embedding_dispatcher.py
import aio_pika
from src.event_driven_rag_service.config import consumer_groups
from src.event_driven_rag_service.infrastructure.event_bus import create_event_log
from src.event_driven_rag_service.tasks.search_tasks import SearchRunTask
from src.event_driven_rag_service.tasks.registry import TASK_ROUTES


class EmbeddingDispatcher:
    """
    Listens to search_query.embedded on the event log.

    search_query.embedded → cpu.search.run

    This event is only emitted for query embeddings (never for chunk embeddings),
    so no discriminator check is needed. Chunk embeddings are terminal — no
    downstream task is dispatched from embedding.completed.

    Single responsibility: trigger search execution once a query is embedded.
    """

    def __init__(self, rmq_connection: aio_pika.Connection):
        self._event_bus = create_event_log()
        self._rmq = rmq_connection

    async def run(self) -> None:
        channel = await self._rmq.channel()
        search_ex = await channel.get_exchange("search")

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
