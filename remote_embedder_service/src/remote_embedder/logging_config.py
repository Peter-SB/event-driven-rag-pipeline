"""Logging setup: console plus a size-bounded log file.

Unlike the rest of the pipeline, this service ships no traces or logs to the
otel-collector — it's a standalone process on the GPU box, so its log file is
the only persistent record of what it did. Bounded to the last
``LOG_MAX_LINES`` lines so a long-running process can't quietly fill the disk.

Kept to INFO and above by default: model load/unload, quantisation fallback,
and request-level warnings/errors are "key info"; per-request HTTP access
lines are not — the client fires ``GET /v1/models`` before *every* batch, and
logging one line per health probe would drown the log budget in routine noise
within hours. Uvicorn's access log is disabled outright rather than merely
downgraded (see ``configure_uvicorn`` below).
"""
from __future__ import annotations

import logging
from pathlib import Path

from remote_embedder.config import Settings

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

# Model-loading libraries are chatty at INFO (download progress, device
# placement, etc.) — not "key info" for this service's own operational log.
_QUIET_LOGGERS = (
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "urllib3",
    "httpx",
    "httpcore",
)


class MaxLinesFileHandler(logging.FileHandler):
    """A FileHandler that keeps at most ``max_lines`` lines on disk.

    The ask is "the last N log lines", not "N files of M bytes each", so this
    trims one file in place rather than juggling ``RotatingFileHandler``'s
    numbered backups. Trimming rewrites the whole file, so it runs every
    ``check_every`` writes rather than on every single one — at
    ``max_lines=10_000`` that's a few hundred KB, cheap enough to do
    occasionally without needing to matter on every log call.
    """

    def __init__(
        self,
        filename: str | Path,
        max_lines: int = 10_000,
        check_every: int = 200,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(filename, mode="a", encoding=encoding)
        self.max_lines = max_lines
        self.check_every = max(1, check_every)
        self._writes_since_check = 0

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self._writes_since_check += 1
        if self._writes_since_check >= self.check_every:
            self._writes_since_check = 0
            self.trim()

    def trim(self) -> None:
        """Drop everything but the last ``max_lines`` lines. Public so tests
        (and callers wanting a deterministic file size, e.g. before shutdown)
        can force a trim without writing hundreds of throwaway records."""
        self.acquire()
        try:
            if self.stream is not None:
                self.stream.flush()
                self.stream.close()
                self.stream = None
            path = Path(self.baseFilename)
            encoding = self.encoding or "utf-8"
            lines = path.read_text(encoding=encoding).splitlines(keepends=True)
            if len(lines) > self.max_lines:
                path.write_text("".join(lines[-self.max_lines :]), encoding=encoding)
            self.stream = self._open()
        finally:
            self.release()


def configure_logging(settings: Settings) -> logging.Handler | None:
    """Attach console + (optionally) bounded file handlers to the root logger.

    Returns the file handler, if one was created, so callers (tests, or a
    graceful-shutdown hook) can force a final trim.
    """
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    # Idempotent: a prior call in the same process (repeated test setup, or a
    # reload) shouldn't accumulate duplicate handlers.
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler: logging.Handler | None = None
    if settings.log_to_file:
        log_path = Path(settings.log_dir) / settings.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = MaxLinesFileHandler(log_path, max_lines=settings.log_max_lines)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return file_handler
