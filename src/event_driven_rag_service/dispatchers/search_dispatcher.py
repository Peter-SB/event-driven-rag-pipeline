# dispatchers/search_dispatcher.py
import aio_pika
from src.event_driven_rag_service.config import consumer_groups
from src.event_driven_rag_service.infrastructure.event_bus import create_event_log
from src.event_driven_rag_service.tasks.embed_task import EmbedTask

# Must match the model used for chunk embeddings so vectors are in the same space.
# If you change this, update ChunkDispatcher._DEFAULT_EMBED_MODEL too.
_QUERY_EMBED_MODEL = "bge-base-v1.5"


class SearchDispatcher:
    """
    Handles both steps of the search pipeline.

    Step 1: search_job.created → gpu.embed.{model}  (embed the query)
    Step 2: search_query.embedded → cpu.search.run  (execute search with the vector)

    Single responsibility: translate search events into tasks. No work is done here.
    """

    def __init__(self, rmq_connection: aio_pika.Connection):
        self._event_bus = create_event_log()
        self._rmq = rmq_connection

    async def run(self) -> None:
        channel = await self._rmq.channel()
        embedding_ex = await channel.get_exchange("embedding")
        search_ex = await channel.get_exchange("search")

        async for event in self._event_bus.subscribe(
            "search_job.created", consumer_group=consumer_groups.SEARCH_JOB_CREATED
        ):
            task = EmbedTask(
                task_type="query",
                model_name=_QUERY_EMBED_MODEL,
                query=event["query"],
                query_job_id=event["query_job_id"],
                trace_id=event.get("trace_id"),
            )
            await embedding_ex.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=f"gpu.embed.{_QUERY_EMBED_MODEL}",
            )
