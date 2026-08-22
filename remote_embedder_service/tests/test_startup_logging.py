"""Startup-time logging: the device this process is actually running on.

The whole point of this service is CUDA embedding. Running on CPU is either a
misconfiguration (DEVICE set wrong, drivers missing, torch built without CUDA
support) or a deliberate dev/test run — either way it should be impossible to
miss in the logs, not something that only shows up as embed calls being
unexpectedly slow.

Tested behaviours
------------------
- Starting the app with device="cpu" logs a WARNING naming the device.
- Starting with a resolved "cuda" device logs no such warning.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from remote_embedder.app import create_app


def test_warns_on_startup_when_device_is_not_cuda(settings, manager, caplog):
    settings.device = "cpu"
    app = create_app(settings=settings, manager=manager)

    with caplog.at_level("WARNING"):
        with TestClient(app):
            pass  # lifespan startup/shutdown runs on context entry/exit

    assert any(
        "cpu" in r.message and "cuda" in r.message.lower() for r in caplog.records
    )


def test_no_device_warning_when_resolved_device_is_cuda(settings, manager, monkeypatch, caplog):
    # Force resolution to "cuda" without needing a real GPU in CI.
    monkeypatch.setattr(type(settings), "resolved_device", lambda self: "cuda")
    app = create_app(settings=settings, manager=manager)

    with caplog.at_level("WARNING"):
        with TestClient(app):
            pass

    assert not any("not CUDA" in r.message for r in caplog.records)
