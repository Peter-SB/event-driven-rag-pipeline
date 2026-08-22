"""Guards against EmbedConfig.local_repo_id pointing at a HF repo that doesn't
actually contain the GGUF file it's paired with (see embedding_config.py:
qwen3_embed_cfg.local_repo_id/model).

This is a lightweight metadata check — it lists files via the HF Hub API, it
never downloads model weights — but it does require network access, so it's
gated behind the `integration` marker like the rest of this directory.

A regression here previously shipped silently: unit tests all run with
MOCK_EMBEDDINGS=1, which short-circuits before `local_repo_id` is ever
resolved against the real repo (see worker/entrypoints/gpu.py:_load_gguf_model).
"""
from __future__ import annotations

import pytest
from huggingface_hub import list_repo_files
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import HFValidationError

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS

pytestmark = pytest.mark.integration

_GGUF_CONFIGS = {
    task_type: cfg for task_type, cfg in EMBED_CONFIGS.items() if cfg.local_repo_id
}


@pytest.mark.parametrize("task_type", sorted(_GGUF_CONFIGS))
def test_local_repo_id_contains_the_gguf_filename(task_type: str):
    """`local_repo_id` must actually host the `model` filename via llama-cpp's
    Llama.from_pretrained(repo_id=local_repo_id, filename=model) — otherwise
    the local GGUF fallback fails every time the remote endpoint is down."""
    cfg = _GGUF_CONFIGS[task_type]

    try:
        files = list_repo_files(cfg.local_repo_id)
    except (HfHubHTTPError, HFValidationError) as exc:
        pytest.fail(
            f"local_repo_id={cfg.local_repo_id!r} for {task_type!r} is not a "
            f"reachable/valid HF repo: {exc}"
        )

    assert cfg.model in files, (
        f"{task_type!r} config expects {cfg.local_repo_id!r} to contain "
        f"{cfg.model!r}, but that repo only has: {files}"
    )
