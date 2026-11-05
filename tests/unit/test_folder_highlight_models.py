"""Tests for the Folder and Highlight data models.

Tested behaviours
-----------------
- CamelCase wire format and snake_case row format both construct the model
- Naive timestamps (SQLite CURRENT_TIMESTAMP) are treated as UTC
- Missing updatedAt falls back to createdAt
- Soft-delete flag and optional fields default correctly
"""
from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from event_driven_rag_service.data_models.folder import Folder
from event_driven_rag_service.data_models.highlight import Highlight


_TS = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Folder
# ---------------------------------------------------------------------------

def test_folder_accepts_camel_case_wire_format():
    folder = Folder(**{"id": 3, "name": "Recipes", "parentId": 1,
                       "createdAt": "2024-06-01T12:00:00.000Z", "updatedAt": "2024-06-02T12:00:00.000Z",
                       "isDeleted": True})
    assert folder.folder_id == 3
    assert folder.parent_id == 1
    assert folder.is_deleted is True
    assert folder.updated_at == datetime(2024, 6, 2, 12, 0, tzinfo=UTC)


def test_folder_accepts_snake_case_row_format():
    folder = Folder(folder_id=3, name="Recipes", created_at=_TS, updated_at=_TS)
    assert folder.folder_id == 3
    assert folder.parent_id is None
    assert folder.is_deleted is False


def test_folder_naive_timestamp_is_treated_as_utc():
    folder = Folder(id=1, name="x", createdAt="2024-06-01 12:00:00", updatedAt="2024-06-01 12:00:00")
    assert folder.created_at == _TS
    assert folder.created_at.tzinfo is not None


def test_folder_missing_updated_at_falls_back_to_created_at():
    folder = Folder(id=1, name="x", createdAt="2024-06-01T12:00:00Z")
    assert folder.updated_at == _TS


def test_folder_null_updated_at_falls_back_to_created_at():
    folder = Folder(id=1, name="x", createdAt="2024-06-01T12:00:00Z", updatedAt=None)
    assert folder.updated_at == _TS


def test_folder_requires_name():
    with pytest.raises(ValidationError):
        Folder(id=1, createdAt=_TS)


# ---------------------------------------------------------------------------
# Highlight
# ---------------------------------------------------------------------------

def test_highlight_accepts_camel_case_wire_format():
    hl = Highlight(**{"id": 7, "postId": 42, "text": "quote", "note": "why", "startOffset": 5,
                      "endOffset": 10, "createdAt": "2024-06-01T12:00:00Z",
                      "updatedAt": "2024-06-01T12:00:00Z", "isDeleted": False})
    assert hl.highlight_id == 7
    assert hl.post_id == 42
    assert (hl.start_offset, hl.end_offset) == (5, 10)


def test_highlight_optional_fields_default():
    hl = Highlight(id=1, postId=2, text="quote", createdAt=_TS)
    assert hl.note is None
    assert hl.start_offset is None
    assert hl.end_offset is None
    assert hl.is_deleted is False
    assert hl.updated_at == _TS


def test_highlight_naive_timestamp_is_treated_as_utc():
    hl = Highlight(id=1, postId=2, text="quote", createdAt="2024-06-01 12:00:00",
                   updatedAt="2024-06-01 12:00:00")
    assert hl.updated_at == _TS


def test_highlight_requires_text():
    with pytest.raises(ValidationError):
        Highlight(id=1, postId=2, createdAt=_TS)
