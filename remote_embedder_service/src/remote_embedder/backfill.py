"""Asks the pipeline to re-queue chunks that never got an embedding.

Models marked ``allow_local=False`` in the pipeline cannot be embedded on the
homelab box, so while this service is down their chunks are skipped and left with
``embedding IS NULL``. This service is the only thing that can finish that work,
so it tells the pipeline when it is up: once at startup, retried every
``retry_s`` until the pipeline answers, then again every ``interval_s`` (12h by
default) in case a batch was skipped while it was running.

The endpoint is idempotent — it only re-queues chunks that are still unembedded —
so calling it too often is harmless.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from collections.abc import Callable

from remote_embedder.config import Settings

logger = logging.getLogger(__name__)

BACKFILL_PATH = "/maintenance/requeue-missing-embeddings"


def _post_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


class BackfillTrigger:
    def __init__(
        self,
        pipeline_url: str,
        interval_s: float,
        retry_s: float,
        timeout_s: float,
        post: Callable[[str, float], dict] = _post_json,
    ) -> None:
        self._url = pipeline_url.rstrip("/") + BACKFILL_PATH
        self._interval_s = interval_s
        self._retry_s = retry_s
        self._timeout_s = timeout_s
        self._post = post
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_settings(cls, cfg: Settings) -> BackfillTrigger | None:
        """None when ``PIPELINE_URL`` is unset, which leaves the feature off."""
        if not cfg.pipeline_url:
            return None
        return cls(
            cfg.pipeline_url,
            cfg.backfill_interval_s,
            cfg.backfill_retry_s,
            cfg.backfill_timeout_s,
        )

    def trigger_once(self) -> bool:
        try:
            result = self._post(self._url, self._timeout_s)
        except Exception as exc:  # network down, pipeline restarting, HTTP error
            logger.warning("Backfill request to %s failed: %s", self._url, exc)
            return False
        logger.info("Backfill requested at %s: %s", self._url, result)
        return True

    def run(self) -> None:
        while True:
            delay = self._interval_s if self.trigger_once() else self._retry_s
            if self._stop.wait(delay):
                return

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.run, name="backfill-trigger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
