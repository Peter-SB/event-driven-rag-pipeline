from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from event_driven_rag_service.data_models.folder import as_utc


class Highlight(BaseModel):
    """A passage the user captured from a post, with an optional note.

    Wire format uses camelCase aliases (matching the sync client).
    DB row format uses snake_case. Both are accepted via populate_by_name=True.
    """
    highlight_id: int = Field(..., alias="id")
    post_id: int = Field(..., alias="postId")
    text: str
    note: Optional[str] = None
    start_offset: Optional[int] = Field(None, alias="startOffset")
    end_offset: Optional[int] = Field(None, alias="endOffset")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")  # Set by the client, used for freshness comparison on upsert
    is_deleted: bool = Field(False, alias="isDeleted")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _default_updated_at(cls, data: object) -> object:
        if isinstance(data, dict) and not data.get("updatedAt") and not data.get("updated_at"):
            fallback = data.get("createdAt") or data.get("created_at")
            if fallback is not None:
                data["updatedAt"] = fallback
        return data

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v: object) -> object:
        return as_utc(v)
