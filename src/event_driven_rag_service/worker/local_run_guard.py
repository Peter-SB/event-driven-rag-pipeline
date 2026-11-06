"""Remembers that a local embedding run was killed, so it is not retried in a loop.

The GPU worker container has a hard memory cap (``mem_limit`` in docker-compose.yml).
If loading or running a model locally goes over it, the kernel kills the whole
process and Docker restarts the container — and RabbitMQ redelivers the same,
never-acked message, which would start the same doomed run again, forever.

A local run therefore drops a marker file before it starts and removes it when it
finishes (successfully or with an ordinary exception). A marker that is still there
when a *new* process first asks to run that model means the previous process died
mid-run. The marker becomes a block, and from then on ``check`` raises
``EmbeddingUnavailableError``. ``EmbedHandler`` treats that as a skip: the task is
acked and the chunks stay ``embedding IS NULL`` for the remote embedder's backfill
(``POST /maintenance/requeue-missing-embeddings``) to pick up.

The files live in the container's writable layer, which survives the automatic
restart but not a recreate — so deploying new code gives the model a fresh chance.
To clear a block by hand: ``docker exec rag-gpu-worker sh -c 'rm /tmp/rag-local-guard/*.blocked'``.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from event_driven_rag_service.exceptions import EmbeddingUnavailableError

logger = logging.getLogger(__name__)


class LocalRunGuard:
    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._active: set[str] = set()  # models this process is running right now
        self._lock = threading.Lock()

    def _path(self, model: str, suffix: str) -> Path:
        return self._dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.{suffix}"

    def check(self, model: str) -> None:
        """Raise ``EmbeddingUnavailableError`` if local runs of *model* are blocked."""
        with self._lock:
            running = self._path(model, "running")
            blocked = self._path(model, "blocked")
            if model not in self._active and running.exists():
                running.replace(blocked)
                logger.error(
                    "Local run of %r was killed before it finished (out of memory?) — "
                    "blocking local embedding for it. Chunks stay unembedded for the "
                    "remote embedder's backfill.",
                    model,
                )
            if blocked.exists():
                raise EmbeddingUnavailableError(
                    f"Local embedding of {model!r} is blocked: a previous local run was "
                    f"killed. Skipping so the chunks wait for the remote embedder. Remove "
                    f"{blocked} or recreate the container to try local again."
                )

    @contextmanager
    def run(self, model: str) -> Iterator[None]:
        """Guard one local load-and-encode. Raises ``EmbeddingUnavailableError`` if blocked."""
        self.check(model)
        running = self._path(model, "running")
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            running.write_text(str(time.time()))
            self._active.add(model)
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(model)
                running.unlink(missing_ok=True)


_default: LocalRunGuard | None = None


def get_guard() -> LocalRunGuard:
    global _default
    if _default is None:
        directory = os.getenv("LOCAL_GUARD_DIR") or Path(tempfile.gettempdir()) / "rag-local-guard"
        _default = LocalRunGuard(directory)
    return _default


def reset_default_guard() -> None:
    """For tests: forget the process-wide guard so the next ``get_guard`` re-reads the env."""
    global _default
    _default = None
