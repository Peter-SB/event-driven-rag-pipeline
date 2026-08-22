"""Service settings, read from the environment.

Mirrors the style of the pipeline's ``config/settings.py``: a frozen-ish dataclass
of ``field(default_factory=...)`` env reads, instantiated once at import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8100")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # File logging. This service has no otel-collector to ship logs to (unlike
    # the rest of the pipeline) — its own log file is the only persistent
    # record on the GPU box, so it's on by default.
    log_to_file: bool = field(default_factory=lambda: _env_bool("LOG_TO_FILE", True))
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "remote_embedder.log"))
    # Lines kept on disk, oldest dropped first. Bounds disk use on a
    # long-running process without needing external log rotation set up.
    log_max_lines: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_LINES", "10000")))

    # Optional shared secret. Empty (the default) means the Authorization header is
    # accepted-and-ignored, which keeps the service drop-in for clients that send a
    # bearer token regardless. Set it to enforce.
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))

    # "cuda", "cpu", or "auto" (cuda when available, else cpu).
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))

    # Texts per forward pass. Matches GpuEmbedWorker.max_batch (32) so a single
    # pipeline batch is normally a single pass; larger inputs are split and rejoined.
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "32")))

    # Seconds a loaded model may sit untouched before the reaper frees its VRAM.
    # 0 disables auto-unload.
    idle_unload_s: float = field(default_factory=lambda: float(os.getenv("IDLE_UNLOAD_S", "900")))
    # How often the reaper thread wakes.
    reaper_interval_s: float = field(
        default_factory=lambda: float(os.getenv("REAPER_INTERVAL_S", "15"))
    )

    # Allow the int8 path at all. Off => every model loads at full/half precision
    # regardless of its registry `quantize` flag (useful on CPU-only boxes).
    enable_quantization: bool = field(
        default_factory=lambda: _env_bool("ENABLE_QUANTIZATION", True)
    )

    # L2-normalise output vectors. The pipeline compares with cosine distance, so
    # normalised vectors are what it expects; LM Studio also returned normalised ones.
    normalize_embeddings: bool = field(
        default_factory=lambda: _env_bool("NORMALIZE_EMBEDDINGS", True)
    )

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch missing or broken — CPU is the safe answer
            return "cpu"


settings = Settings()
