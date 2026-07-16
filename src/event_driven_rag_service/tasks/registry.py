"""
Task registry — single source of truth for task routing and deserialization.

Usage
-----
Dispatchers — look up where to publish a task:

    task  = ChunkTask(task_type="body", post_id=1, post_table="posts")
    route = TASK_ROUTES[task.kind]
    exchange = await channel.get_exchange(route.exchange)
    await exchange.publish(
        aio_pika.Message(task.model_dump_json().encode()),
        # Static routing key for single-destination routes:
        routing_key=route.routing_key,
        # Config-dependent routes (embed, infer) instead resolve the key
        # from the relevant config object, e.g. EMBED_CONFIGS[task_type].queue.
    )

Workers — deserialize an incoming payload to the correct typed model:

    task = parse_task(json.loads(message.body))
    if isinstance(task, ChunkTask):
        ...
    elif isinstance(task, EmbedTask):
        ...

Adding a new task type
----------------------
1. Create a new subclass of BaseTask with a unique ``kind`` Literal.
2. Add an entry to TASK_ROUTES.
3. Add the new class to the AnyTask union.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Union

from pydantic import Field, TypeAdapter

from .chunk_task import ChunkTask
from .embed_task import EmbedTask
from .infer_task import InferTask
from .search_tasks import SearchRankTask, SearchRunTask


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskRoute:
    """Exchange (and, for fixed-destination routes, routing key) for one task kind.

    ``routing_key`` is a static value for routes with one destination queue
    (chunk, search_run, search_rank). It's left unset for routes where the
    destination queue depends on task-specific config (embed, infer) — those
    dispatchers must resolve the actual key from the relevant config object's
    ``.queue`` field (e.g. ``EMBED_CONFIGS[task_type].queue``) at publish
    time. Deriving a routing key from a model *name* instead of that config's
    ``.queue`` is a known footgun: multiple task_types can share one physical
    queue under different model strings (see chunk_dispatcher.py), so a
    name-derived key silently targets a queue nothing is bound to and
    RabbitMQ drops the message with no error.
    """
    exchange: str
    routing_key: str | None = None


#: Maps each ``kind`` value to its publish destination.
#: Dispatchers use this instead of hard-coding exchange/routing-key strings.
TASK_ROUTES: dict[str, TaskRoute] = {
    "chunk":       TaskRoute(exchange="ingestion", routing_key="cpu.chunk.post"),
    "embed":       TaskRoute(exchange="embedding"),
    "infer":       TaskRoute(exchange="inference"),
    "search_run":  TaskRoute(exchange="search",    routing_key="cpu.search.run"),
    "search_rank": TaskRoute(exchange="search",    routing_key="cpu.search.rank"),
}


# ---------------------------------------------------------------------------
# Discriminated-union parser for workers
# ---------------------------------------------------------------------------

#: All concrete task types.  Extend the Union when adding new task classes.
AnyTask = Annotated[
    Union[ChunkTask, EmbedTask, InferTask, SearchRunTask, SearchRankTask],
    Field(discriminator="kind"),
]

_adapter: TypeAdapter[AnyTask] = TypeAdapter(AnyTask)


def parse_task(payload: dict) -> AnyTask:
    """Deserialise a raw dict (from RabbitMQ message body) to the correct typed task.

    Uses the ``kind`` discriminator field to select the concrete model.
    Raises ``pydantic.ValidationError`` on malformed payloads so the caller
    can nack the message and route it to the DLQ.
    """
    return _adapter.validate_python(payload)
