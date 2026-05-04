from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class BaseTask(BaseModel):
    """Base class for all tasks dispatched via RabbitMQ.

    Subclasses must declare:
        kind: Literal["<name>"] = "<name>"

    The ``kind`` field is the discriminator used by TaskRegistry.parse_task()
    to deserialise incoming RabbitMQ messages to the correct typed model.
    """

    # Unique per instance — class-level declaration ensures each task gets its
    # own UUID rather than sharing one across all instances.
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    kind: str  # Literal value overridden by each subclass

    # Distributed tracing — propagated from the triggering event so the full
    # chain (event → task → new event) can be reconstructed in logs.
    trace_id: Optional[str] = None
    source_event_id: Optional[str] = None  # event_id of the event that triggered this task
