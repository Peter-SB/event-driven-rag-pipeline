"""The embedder asks the pipeline to re-queue unembedded chunks.

Tested behaviours
------------------
- Off unless PIPELINE_URL is set.
- Posts to <PIPELINE_URL>/maintenance/requeue-missing-embeddings, however the base URL ends.
- Retries on the short delay until the pipeline answers, then settles on the long interval.
- The app starts it on startup and stops it on shutdown.
"""
from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from remote_embedder.app import create_app
from remote_embedder.backfill import BACKFILL_PATH, BackfillTrigger

INTERVAL_S = 43200
RETRY_S = 60


class StopAfter:
    """Stands in for threading.Event: records each wait() delay, "stops" after n of them."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.delays: list[float] = []

    def wait(self, delay: float) -> bool:
        self.delays.append(delay)
        return len(self.delays) >= self.n

    def set(self) -> None:
        self.n = 0


def make_trigger(post) -> BackfillTrigger:
    return BackfillTrigger("http://pipeline:8000/", INTERVAL_S, RETRY_S, 5.0, post=post)


def test_disabled_without_pipeline_url(settings):
    settings.pipeline_url = ""
    assert BackfillTrigger.from_settings(settings) is None


def test_posts_to_the_maintenance_endpoint(settings):
    settings.pipeline_url = "http://pipeline:8000/"
    trigger = BackfillTrigger.from_settings(settings)
    urls = []
    trigger._post = lambda url, timeout: urls.append(url) or {}

    assert trigger.trigger_once() is True
    assert urls == ["http://pipeline:8000" + BACKFILL_PATH]


def test_retries_quickly_until_it_succeeds_then_waits_the_full_interval():
    results = iter([ConnectionError("down"), ConnectionError("down"), {"requeued_chunks": 3}])

    def post(url, timeout):
        outcome = next(results)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    trigger = make_trigger(post)
    trigger._stop = StopAfter(3)
    trigger.run()

    assert trigger._stop.delays == [RETRY_S, RETRY_S, INTERVAL_S]


def test_app_starts_and_stops_the_trigger(settings, manager):
    called = threading.Event()
    trigger = make_trigger(lambda url, timeout: called.set() or {})
    app = create_app(settings=settings, manager=manager, backfill=trigger)

    with TestClient(app):
        assert called.wait(timeout=5)

    assert trigger._thread is None  # stop() joined and cleared it
