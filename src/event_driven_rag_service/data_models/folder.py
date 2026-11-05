from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def as_utc(v: object) -> object:
    """The app's SQLite timestamps (CURRENT_TIMESTAMP) arrive without a timezone.
    Treat naive values as UTC so they compare safely against stored TIMESTAMPTZ."""
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return v
    if isinstance(v, datetime) and v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    return v


class Folder(BaseModel):
    """A user-created folder used to organise posts, with optional nesting.

    Wire format uses camelCase aliases (matching the sync client).
    DB row format uses snake_case. Both are accepted via populate_by_name=True.

    Post membership is not stored here: it already travels on
    ``Post.folder_ids``, which resolve to ``folder_id`` rows in this table.
    """
    folder_id: int = Field(..., alias="id")
    name: str
    parent_id: Optional[int] = Field(None, alias="parentId")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")  # Set by the client, used for freshness comparison on upsert
    is_deleted: bool = Field(False, alias="isDeleted")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _default_updated_at(cls, data: object) -> object:
        """Folders created before the client tracked updatedAt fall back to createdAt."""
        if isinstance(data, dict) and not data.get("updatedAt") and not data.get("updated_at"):
            fallback = data.get("createdAt") or data.get("created_at")
            if fallback is not None:
                data["updatedAt"] = fallback
        return data

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v: object) -> object:
        return as_utc(v)
