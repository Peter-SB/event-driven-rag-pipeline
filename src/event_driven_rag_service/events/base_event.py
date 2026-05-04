
from __future__ import annotations

import uuid
from datetime import datetime, UTC
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict

class BaseEvent(BaseModel):
    """
    Base model for all events in the system.
        
    Events should contain enough data to be meaningful and replayable, but not so much that they become a database.

    What to include in events:
    - Identifiers (always)
    - Key fields relevant to the event
    - Versioning / metadata
    - Optionally: derived/lightweight data
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())) # todo: check to use unique id for impodence
    event_type: str
    event_version: int = 1

    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Distributed tracing
    trace_id: Optional[str] = None

    # Optional correlation (group related events) todo: comeback to and decide if we need
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    model_config = ConfigDict(extra="forbid") # ??

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")