"""Sanity checks that every embedding queue in EMBED_CONFIGS is declared in BINDINGS."""
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.task_queue import BINDINGS


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
