"""Sanity checks that every embedding queue in EMBED_CONFIGS is declared in BINDINGS."""
from unittest.mock import patch

import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS, EmbedConfig
from event_driven_rag_service.infrastructure.task_queue import BINDINGS, verify_embedding_topology


def _embedding_queues() -> set[str]:
    return {queue for _, queue in BINDINGS["embedding"]}


def test_every_embed_config_queue_is_declared():
    declared = _embedding_queues()
    for task_type, cfg in EMBED_CONFIGS.items():
        assert cfg.queue in declared, (
            f"EMBED_CONFIGS[{task_type!r}].queue={cfg.queue!r} has no matching entry in BINDINGS['embedding']"
        )


def test_no_undeclared_queues_in_embed_configs():
    configured = {cfg.queue for cfg in EMBED_CONFIGS.values()}
    declared = _embedding_queues()
    orphans = declared - configured
    assert not orphans, f"Queues declared in BINDINGS but not used by any EMBED_CONFIGS: {orphans}"


# ---------------------------------------------------------------------------
# verify_embedding_topology() — the runtime startup guard
#
# This is the loud-failure counterpart to the two tests above: it's called at
# process startup (API, dispatcher, GPU worker entrypoints) so a deploy with
# EMBED_CONFIGS/BINDINGS drift crashes immediately instead of silently
# dropping embed tasks the way the original summary_title bug did.
# ---------------------------------------------------------------------------

def test_verify_embedding_topology_passes_for_current_config():
    verify_embedding_topology()  # must not raise


def test_verify_embedding_topology_raises_when_a_configured_queue_is_undeclared():
    stale_configs = dict(EMBED_CONFIGS)
    stale_configs["summary_title"] = EmbedConfig(
        model="some-new-model", queue="gpu.embed.some-new-model-queue", dim=1024
    )
    with patch(
        "event_driven_rag_service.infrastructure.task_queue.EMBED_CONFIGS", stale_configs
    ):
        with pytest.raises(RuntimeError, match="gpu.embed.some-new-model-queue"):
            verify_embedding_topology()
