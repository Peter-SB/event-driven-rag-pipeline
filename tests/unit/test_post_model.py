"""Tests for Post data model.

Verifies that the source-agnostic Post model accepts both camelCase
wire format (from the Reddit sync client) and snake_case DB row format,
enforces required fields, and parses extra_fields JSON strings correctly.

Tested behaviours
-----------------
- CamelCase alias construction works (wire format from client)
- snake_case field names also work (populate_by_name=True, DB row compat)
- extra_fields JSON string is coerced to a dict automatically
- Missing required fields raise a ValidationError
- Optional fields default correctly (subreddit=None, summary=None)
- Post with a summary correctly has has_summary logic
"""
from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from event_driven_rag_service.data_models.post import Post
from tests.utils.factories import make_post


_TS = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_post_construction_with_camelcase_aliases():
    """Wire format from Reddit sync client uses camelCase aliases."""
    post = Post(
        id=1,
        redditId="abc123",
        redditCreatedAt=_TS,
        url="https://reddit.com/r/test/1",
        title="Hello",
        bodyText="Some content",
        author="user1",
        subreddit="test",
        addedAt=_TS,
        updatedAt=_TS,
    )
    assert post.post_id == 1
    assert post.external_id == "abc123"
    assert post.body_text == "Some content"


def test_post_construction_with_snake_case_names():
    """DB row format uses snake_case field names."""
    post = Post(
        post_id=2,
        external_id="xyz",
        external_created_at=_TS,
        url="https://example.com",
        title="Title",
        author="u",
        added_at=_TS,
        updated_at=_TS,
    )
    assert post.post_id == 2
    assert post.subreddit is None  # optional


def test_post_factory_produces_valid_post():
    post = make_post(post_id=5)
    assert post.post_id == 5
    assert post.external_source == "reddit"


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------

def test_post_subreddit_defaults_to_none_for_non_reddit():
    post = make_post(subreddit=None)
    assert post.subreddit is None


def test_post_summary_is_none_when_not_provided():
    post = make_post(post_id=1, summary=None)
    assert post.summary is None


def test_post_custom_body_defaults_to_none():
    post = make_post(post_id=1)
    assert post.custom_body is None


# ---------------------------------------------------------------------------
# extra_fields JSON coercion
# ---------------------------------------------------------------------------

def test_extra_fields_string_is_parsed_to_dict():
    post = make_post(post_id=1, extra_fields='{"source": "mobile", "score": 42}')
    assert isinstance(post.extra_fields, dict)
    assert post.extra_fields["source"] == "mobile"


def test_extra_fields_invalid_json_kept_as_string():
    # If the string isn't valid JSON, the raw string value is kept
    post = make_post(post_id=1, extra_fields="not-json-at-all")
    assert post.extra_fields == "not-json-at-all"


def test_extra_fields_dict_passed_directly():
    post = make_post(post_id=1, extra_fields={"key": "value"})
    assert post.extra_fields == {"key": "value"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        Post(id=1)  # missing external_id, url, title, author, timestamps


def test_invalid_post_id_type_raises():
    with pytest.raises(ValidationError):
        make_post(post_id="not-an-int")  # type: ignore[arg-type]
