"""Model swaps and the GPU queue.

Three things went wrong together on the 8 GB card: a failed quantised load
quietly reloaded at full precision (25x slower, timing the caller out), requests
for the GPU waited on a bare lock in no particular order, and a backlog just kept
growing. These pin the replacements. Fake loaders and models; no GPU.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from remote_embedder import manager as manager_module
from remote_embedder.app import create_app
from remote_embedder.manager import (
    PRIORITY_EMBED,
    PRIORITY_RERANK,
    ModelLoadError,
    ModelManager,
    QueueFullError,
    _looks_like_out_of_memory,
    _TurnQueue,
)

from conftest import RecordingLoader

BGE_BASE = "text-embedding-bge-base-en-v1.5"
RERANK = "text-rerank-qwen3-reranker-0.6b"

OOM = RuntimeError(
    "Some modules are dispatched on the CPU or the disk. Make sure you have "
    "enough GPU RAM to fit the quantized model."
)


# --- the queue -------------------------------------------------------------


def _hold_the_gpu(queue: _TurnQueue):
    """Take a turn on a thread and keep it until released. Returns the release event."""
    holding, release = threading.Event(), threading.Event()

    def run():
        with queue.turn(PRIORITY_EMBED):
            holding.set()
            release.wait(5)

    threading.Thread(target=run, daemon=True).start()
    assert holding.wait(5)
    return release


def _join_the_line(queue: _TurnQueue, name: str, priority: int, order: list[str]):
    def run():
        with queue.turn(priority):
            order.append(name)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while queue.waiting < _join_the_line.expected and time.monotonic() < deadline:
        time.sleep(0.005)
    return t


def _run_line(entries: list[tuple[str, int]], max_waiting: int = 16) -> list[str]:
    queue = _TurnQueue(max_waiting)
    release = _hold_the_gpu(queue)
    order: list[str] = []
    threads = []
    for i, (name, priority) in enumerate(entries, start=1):
        _join_the_line.expected = i
        threads.append(_join_the_line(queue, name, priority, order))
    release.set()
    for t in threads:
        t.join(5)
    return order


def test_requests_run_in_the_order_they_arrived():
    assert _run_line([("a", PRIORITY_EMBED), ("b", PRIORITY_EMBED), ("c", PRIORITY_EMBED)]) == [
        "a",
        "b",
        "c",
    ]


def test_a_rerank_goes_ahead_of_embeds_that_arrived_first():
    order = _run_line(
        [("embed-1", PRIORITY_EMBED), ("embed-2", PRIORITY_EMBED), ("rerank", PRIORITY_RERANK)]
    )
    assert order == ["rerank", "embed-1", "embed-2"]


def test_a_full_line_refuses_new_arrivals_instead_of_queueing_them():
    queue = _TurnQueue(max_waiting=1)
    release = _hold_the_gpu(queue)
    order: list[str] = []
    _join_the_line.expected = 1
    waiter = _join_the_line(queue, "waiting", PRIORITY_EMBED, order)

    with pytest.raises(QueueFullError):
        with queue.turn(PRIORITY_EMBED):
            pass

    release.set()
    waiter.join(5)
    assert order == ["waiting"]


def test_an_idle_gpu_never_refuses_even_with_a_zero_length_line():
    queue = _TurnQueue(max_waiting=0)
    with queue.turn(PRIORITY_EMBED):
        pass


# --- swapping models ---------------------------------------------------------


class _OomThenFine:
    """Loader that runs out of memory `failures` times before it works."""

    def __init__(self, failures: int, error: Exception = OOM) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0
        self._inner = RecordingLoader()

    def __call__(self, spec, settings):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self._inner(spec, settings)


@pytest.fixture
def quick_retry(settings):
    settings.load_retry_wait_s = 0
    return settings


def test_an_out_of_memory_load_is_retried_once_after_clearing_the_gpu(quick_retry, monkeypatch):
    cleared = []
    monkeypatch.setattr(ModelManager, "_clear_gpu_memory", staticmethod(lambda: cleared.append(1)))
    loader = _OomThenFine(failures=1)
    mgr = ModelManager(quick_retry, loader=loader)

    mgr.load(BGE_BASE)

    assert loader.calls == 2
    assert mgr.loaded_model_id == BGE_BASE
    assert cleared, "the retry should follow a hard memory clear"


def test_two_out_of_memory_loads_fail_with_a_clear_error_not_a_slow_fallback(quick_retry):
    loader = _OomThenFine(failures=5)
    mgr = ModelManager(quick_retry, loader=loader)

    with pytest.raises(ModelLoadError, match="not enough free GPU memory"):
        mgr.load(BGE_BASE)

    assert loader.calls == 2
    assert mgr.loaded_model_id is None


def test_a_failure_that_memory_cannot_fix_is_not_retried(quick_retry):
    loader = _OomThenFine(failures=5, error=OSError("network is down"))
    mgr = ModelManager(quick_retry, loader=loader)

    with pytest.raises(ModelLoadError, match="network is down"):
        mgr.load(BGE_BASE)

    assert loader.calls == 1


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        type("OutOfMemoryError", (RuntimeError,), {})("boom"),
        OOM,
    ],
)
def test_out_of_memory_errors_are_recognised(error):
    assert _looks_like_out_of_memory(error)


def test_other_errors_are_not_mistaken_for_out_of_memory():
    assert not _looks_like_out_of_memory(OSError("network is down"))


def test_swapping_models_clears_gpu_memory_before_the_next_load(settings, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        ModelManager, "_clear_gpu_memory", staticmethod(lambda: events.append("clear"))
    )
    inner = RecordingLoader()

    def loader(spec, s):
        events.append(f"load {spec.id}")
        return inner(spec, s)

    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE)
    events.clear()
    mgr.load(RERANK)

    assert events == ["clear", f"load {RERANK}"]


# --- no silent full-precision fallback --------------------------------------


def test_a_failed_quantised_embedder_load_on_cuda_raises_instead_of_loading_full_size(
    settings, monkeypatch
):
    import sentence_transformers

    calls: list[dict] = []

    class FakeSentenceTransformer:
        def __init__(self, repo_id, **kwargs):
            calls.append(kwargs)
            if "model_kwargs" in kwargs:
                raise OOM
            raise AssertionError("must not fall back to a full-precision load")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    settings.device = "cuda"
    settings.enable_quantization = True
    spec = manager_module.get_spec("text-embedding-qwen3-embedding-4b")

    with pytest.raises(RuntimeError, match="GPU RAM"):
        manager_module._load_embedder(spec, settings)

    assert len(calls) == 1 and "model_kwargs" in calls[0]


def test_a_failed_int4_reranker_load_on_cuda_raises_instead_of_loading_full_size(
    settings, monkeypatch
):
    import transformers

    calls: list[dict] = []

    class FakeCausalLM:
        @staticmethod
        def from_pretrained(repo_id, **kwargs):
            calls.append(kwargs)
            raise OOM

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeCausalLM)
    settings.device = "cuda"
    settings.enable_quantization = True
    spec = manager_module.get_spec("text-rerank-qwen3-reranker-4b")

    with pytest.raises(RuntimeError, match="GPU RAM"):
        manager_module._load_reranker(spec, settings)

    assert len(calls) == 1 and "quantization_config" in calls[0]


# --- the HTTP face of a full line --------------------------------------------


def test_a_full_queue_answers_503_with_retry_after(settings):
    class Busy(ModelManager):
        def encode(self, model_id, texts):
            raise QueueFullError("16 requests are already waiting for the GPU")

    app = create_app(settings=settings, manager=Busy(settings, loader=RecordingLoader()))
    with TestClient(app) as client:
        resp = client.post("/v1/embeddings", json={"model": BGE_BASE, "input": ["hello"]})

    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "10"
    assert "busy" in resp.json()["detail"]
