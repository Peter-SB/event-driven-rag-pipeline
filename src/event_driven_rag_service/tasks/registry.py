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
        routing_key=route.resolve_key(task),
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

from .base_task import BaseTask
from .chunk_task import ChunkTask
from .embed_task import EmbedTask
from .infer_task import InferTask
from .search_tasks import SearchRankTask, SearchRunTask


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskRoute:
    """Exchange and routing-key config for publishing one task kind.

    ``routing_key`` may contain Python format placeholders that are
    interpolated from the task's own fields, e.g. ``"gpu.embed.{model_name}"``.
    """
    exchange: str
    routing_key: str

    def resolve_key(self, task: BaseTask) -> str:
        """Substitute task fields into the routing key template."""
        data = task.model_dump()
        # Strip HuggingFace org prefix so "BAAI/bge-base-en-v1.5" routes to
        # "gpu.embed.bge-base-en-v1.5", not "gpu.embed.BAAI/bge-base-en-v1.5".
        if "model_name" in data and isinstance(data["model_name"], str):
            data["model_name"] = data["model_name"].split("/")[-1].lower()
        return self.routing_key.format_map(data)


#: Maps each ``kind`` value to its publish destination.
#: Dispatchers use this instead of hard-coding exchange/routing-key strings.
TASK_ROUTES: dict[str, TaskRoute] = {
    "chunk":       TaskRoute(exchange="ingestion", routing_key="cpu.chunk.post"),
    "embed":       TaskRoute(exchange="embedding", routing_key="gpu.embed.{model_name}"),
    "infer":       TaskRoute(exchange="inference", routing_key="gpu.infer_local.{model}"),
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
