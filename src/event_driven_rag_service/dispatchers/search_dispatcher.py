# dispatchers/search_dispatcher.py
import asyncio
import logging
from datetime import datetime, UTC

import aio_pika

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.config.reranking_config import RERANK_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import set_queue_lag
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.tasks.registry import TASK_ROUTES
from event_driven_rag_service.tasks.search_tasks import (
    CandidateChunkRef,
    SearchFinalizeTask,
    SearchRankTask,
)

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


# Map model name → EmbedConfig so the dispatcher can route by embedding_profile.
_MODEL_TO_CONFIG = {cfg.primary: cfg for cfg in EMBED_CONFIGS.values()}


#: Reverse map for rerank routing, mirroring _MODEL_TO_CONFIG above.
_PROFILE_TO_RERANK_CONFIG = RERANK_CONFIGS


class SearchDispatcher:
    """
    Handles three steps of the search pipeline, run concurrently.

    Step 1: search_job.created         → gpu.embed.{model}     (embed the query)
    Step 2: search_candidates.retrieved → gpu.rerank.{model}    (rerank=True)
                                        → cpu.search.finalize   (rerank=False)

    (search_query.embedded → cpu.search.run is EmbeddingDispatcher's hop —
    see embedding_dispatcher.py.)

    Single responsibility: translate search events into tasks. No work is done here.
    """

    def __init__(self, rmq_connection: aio_pika.Connection, event_bus: EventBusBase) -> None:
        self._event_bus = event_bus
        self._rmq = rmq_connection

    async def run(self) -> None:
        await asyncio.gather(
            self._run_job_created(),
            self._run_candidates_retrieved(),
        )

    async def _run_job_created(self) -> None:
        channel = await self._rmq.channel()
        embedding_ex = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)

        async for event in self._event_bus.subscribe(
            "search_job.created", consumer_group=consumer_groups.SEARCH_JOB_CREATED
        ):
            _record_queue_lag(event, "search")
            # Use the model that embedded the chunks so dimensions match at search time.
            profile = event["embedding_profile"]
            embed_cfg = _MODEL_TO_CONFIG.get(profile)
            if embed_cfg is None:
                logger.error(
                    "SearchDispatcher: unknown embedding_profile %r for job %s — skipping",
                    profile, event.get("query_job_id"),
                )
                continue
            task = EmbedTask(
                task_type="query",
                model_name=embed_cfg.primary,
                query=event["query"],
                query_job_id=event["query_job_id"],
                trace_id=event.get("trace_id"),
            )
            await embedding_ex.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=embed_cfg.queue,
            )
            logger.debug(
                "SearchDispatcher: dispatched query embed task for job_id=%s",
                event.get("query_job_id"),
            )

    async def _run_candidates_retrieved(self) -> None:
        channel = await self._rmq.channel()
        embedding_ex = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
        search_ex = await channel.declare_exchange("search", aio_pika.ExchangeType.TOPIC, durable=True)

        async for event in self._event_bus.subscribe(
            "search_candidates.retrieved",
            consumer_group=consumer_groups.SEARCH_CANDIDATES_RETRIEVED,
        ):
            _record_queue_lag(event, "search")
            job_id = event["query_job_id"]
            candidates = [CandidateChunkRef(**c) for c in event["candidates"]]

            if event.get("rerank"):
                profile = event.get("rerank_profile") or "default"
                rerank_cfg = _PROFILE_TO_RERANK_CONFIG.get(profile)
                if rerank_cfg is None:
                    logger.error(
                        "SearchDispatcher: unknown rerank_profile %r for job %s — skipping",
                        profile, job_id,
                    )
                    continue
                task = SearchRankTask(
                    job_id=job_id,
                    candidates=candidates,
                    rerank_profile=profile,
                    trace_id=event.get("trace_id"),
                )
                await embedding_ex.publish(
                    aio_pika.Message(task.model_dump_json().encode()),
                    routing_key=rerank_cfg.queue,
                )
                logger.debug(
                    "SearchDispatcher: dispatched rerank task for job_id=%s (profile=%s)",
                    job_id, profile,
                )
            else:
                route = TASK_ROUTES["search_finalize"]
                task = SearchFinalizeTask(
                    job_id=job_id,
                    candidates=candidates,
                    trace_id=event.get("trace_id"),
                )
                await search_ex.publish(
                    aio_pika.Message(task.model_dump_json().encode()),
                    routing_key=route.routing_key,
                )
                logger.debug(
                    "SearchDispatcher: dispatched finalize task for job_id=%s",
                    job_id,
                )
