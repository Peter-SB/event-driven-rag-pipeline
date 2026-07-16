"""Unit tests for chunk table name parsing utilities.

No I/O.  Tests that build_chunk_table_suffix_map and parse_chunk_table_name
correctly round-trip all entries in EMBED_CONFIGS and handle edge cases.
"""
from __future__ import annotations

import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS, EmbedConfig
from event_driven_rag_service.utils.build_table_names import (
    build_chunk_table_name,
    build_chunk_table_suffix_map,
    parse_chunk_table_name,
)


# ---------------------------------------------------------------------------
# build_chunk_table_suffix_map
# ---------------------------------------------------------------------------

def test_suffix_map_has_one_entry_per_config():
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)
    # Each task_type produces one suffix key (two task types may share a model,
    # but their task_type prefix makes the suffix unique).
    assert len(suffix_map) == len(EMBED_CONFIGS)


def test_suffix_map_values_are_task_type_and_cfg():
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)
    for suffix, (task_type, cfg) in suffix_map.items():
        assert task_type in EMBED_CONFIGS
        assert EMBED_CONFIGS[task_type] is cfg


def test_suffix_map_uses_custom_embed_configs():
    custom = {
        "my_type": EmbedConfig(model="my-model-v1.0", queue="gpu.embed.my", dim=128),
    }
    suffix_map = build_chunk_table_suffix_map(custom)
    assert "my_type_my_model_v1_0" in suffix_map


def test_suffix_map_defaults_to_embed_configs_when_none_given():
    suffix_map_default = build_chunk_table_suffix_map()
    suffix_map_explicit = build_chunk_table_suffix_map(EMBED_CONFIGS)
    assert set(suffix_map_default.keys()) == set(suffix_map_explicit.keys())


# ---------------------------------------------------------------------------
# parse_chunk_table_name — round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_type", list(EMBED_CONFIGS.keys()))
def test_round_trip_all_task_types(task_type: str):
    """Every EMBED_CONFIGS entry round-trips through build + parse."""
    cfg = EMBED_CONFIGS[task_type]
    table = build_chunk_table_name("posts_main", task_type, cfg.model)
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)

    result = parse_chunk_table_name(table, suffix_map)

    assert result is not None
    post_table, parsed_type, parsed_cfg = result
    assert post_table == "posts_main"
    assert parsed_type == task_type
    assert parsed_cfg is cfg


def test_round_trip_library_with_underscore():
    """Library IDs containing underscores parse correctly."""
    cfg = EMBED_CONFIGS["body"]
    table = build_chunk_table_name("posts_my_lib", "body", cfg.model)
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)

    result = parse_chunk_table_name(table, suffix_map)

    assert result is not None
    post_table, task_type, _ = result
    assert post_table == "posts_my_lib"
    assert task_type == "body"


# ---------------------------------------------------------------------------
# parse_chunk_table_name — invalid inputs
# ---------------------------------------------------------------------------

def test_returns_none_for_table_without_chunks_marker():
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)
    assert parse_chunk_table_name("posts_main", suffix_map) is None


def test_returns_none_for_unknown_suffix():
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)
    assert parse_chunk_table_name("posts_main_chunks_ghost_model", suffix_map) is None


def test_returns_none_for_empty_string():
    suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)
    assert parse_chunk_table_name("", suffix_map) is None
