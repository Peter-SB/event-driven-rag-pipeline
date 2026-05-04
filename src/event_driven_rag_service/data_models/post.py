import json
from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Post(BaseModel):
    """Canonical data model for a post from any external source.

    Designed to be source-agnostic. Reddit-specific fields (subreddit) are
    nullable so the model works for any future content source.

    Wire format uses camelCase aliases (matching the Reddit sync client).
    DB row format uses snake_case. Both are accepted via populate_by_name=True.
    """

    post_id: int = Field(..., alias="id")

    # Source identity — universal fields present for every source
    external_id: str = Field(..., alias="redditId")          # source-specific ID; alias kept for Reddit client compat
    external_source: str = Field("reddit", alias="externalSource")
    external_created_at: datetime = Field(..., alias="redditCreatedAt")  # alias kept for Reddit client compat

    url: str
    title: str
    body_text: Optional[str] = Field(None, alias="bodyText")
    author: str

    # Reddit-only — nullable for non-Reddit sources
    subreddit: Optional[str] = None

    added_at: datetime = Field(..., alias="addedAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    custom_title: Optional[str] = Field(None, alias="customTitle")
    custom_body: Optional[str] = Field(None, alias="customBody")
    notes: Optional[str] = None
    rating: Optional[float] = None
    is_read: bool = Field(False, alias="isRead")
    read_at: Optional[datetime] = Field(None, alias="readAt")
    is_favorite: bool = Field(False, alias="isFavorite")
    is_archived: bool = Field(False, alias="isArchived")
    queued_at: Optional[datetime] = Field(None, alias="queuedAt")
    is_deleted: bool = Field(False, alias="isDeleted")
    folder_ids: list[int] = Field(default_factory=list, alias="folderIds")
    extra_fields: Optional[Union[dict, str]] = Field(None, alias="extraFields")
    body_min_hash: Optional[str] = Field(None, alias="bodyMinHash")
    summary: Optional[str] = None
    embedded_at: Optional[datetime] = Field(None, alias="embeddedAt")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("extra_fields", mode="before")
    @classmethod
    def _coerce_extra_fields(cls, v: object) -> object:
        """Parse JSON strings into dicts so callers never see raw JSON."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v

# Old come back to
    # uuid: str
    # post_id: int
    # external_id: str 
    # external_source: str 
    # url: str
    # title: str
    # custom_title: Optional[str] = Field(None)
    # body_text: Optional[str] = Field(None) 
    # custom_body: Optional[str] = Field(None)
    # author: str
    # subreddit: str
    # external_created_at: datetime 
    # added_at: datetime
    # updated_at: datetime
    # notes: Optional[str] = None
    # rating: Optional[float] = None
    # is_read: bool = Field(False)
    # read_at: Optional[datetime] = Field(None)
    # is_favorite: bool = Field(False)
    # is_archived: bool = Field(False)
    # queued_at: Optional[datetime] = Field(None)
    # is_deleted: bool = Field(False)
    # folder_ids: list[int] = Field(default_factory=list)
    # extra_fields: Optional[Union[dict, str]] = Field(None)
    # body_min_hash: Optional[str] = Field(None)
    # summary: Optional[str] = None

    # model_config = ConfigDict(populate_by_name=True)