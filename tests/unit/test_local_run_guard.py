"""A local embedding run that gets the process killed is not retried when the task is redelivered.

The GPU worker container has a memory cap; going over it kills the whole process and
Docker restarts it with RabbitMQ redelivering the same unacked message. Without the
guard that loops forever on one model load.

Tested behaviours
------------------
- A run that finishes, or raises an ordinary exception, leaves no trace and never blocks.
- A run that is "killed" (marker left behind) blocks local runs in the next process:
  the next check raises EmbeddingUnavailableError, which EmbedHandler turns into a skip
  so the chunks wait for the remote embedder's backfill.
- The block survives further restarts, and is per model.
- _LazyLocalModel (the real local leg) honours the block without loading the model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

os.environ["MOCK_EMBEDDINGS"] = "1"

from event_driven_rag_service.exceptions import EmbeddingUnavailableError
from event_driven_rag_service.worker import local_run_guard
from event_driven_rag_service.worker.entrypoints import gpu as gpu_entrypoint
from event_driven_rag_service.worker.local_run_guard import LocalRunGuard

MODEL = "Qwen/Qwen3-Embedding-4B"


def simulate_killed_run(directory) -> None:
    """Leave behind what a run killed mid-way leaves: its marker, with no cleanup.

    Written directly rather than by abandoning a run() — Python would tidy up an
    abandoned context manager, which a real SIGKILL does not.
    """
    guard = LocalRunGuard(directory)
    guard._dir.mkdir(parents=True, exist_ok=True)
    guard._path(MODEL, "running").write_text("0")


def test_finished_run_leaves_nothing_and_does_not_block(tmp_path):
    guard = LocalRunGuard(tmp_path)
    with guard.run(MODEL):
        pass
    LocalRunGuard(tmp_path).check(MODEL)  # a fresh process sees no problem
    assert list(tmp_path.iterdir()) == []


def test_ordinary_exception_does_not_block(tmp_path):
    guard = LocalRunGuard(tmp_path)
    with pytest.raises(MemoryError):
        with guard.run(MODEL):
            raise MemoryError  # Python-level failure: the process survived
    LocalRunGuard(tmp_path).check(MODEL)


def test_killed_run_blocks_the_next_process(tmp_path):
    simulate_killed_run(tmp_path)

    restarted = LocalRunGuard(tmp_path)
    with pytest.raises(EmbeddingUnavailableError):
        restarted.check(MODEL)


def test_block_persists_across_further_restarts(tmp_path):
    simulate_killed_run(tmp_path)
    with pytest.raises(EmbeddingUnavailableError):
        LocalRunGuard(tmp_path).check(MODEL)

    with pytest.raises(EmbeddingUnavailableError):
        LocalRunGuard(tmp_path).check(MODEL)  # and the run() path refuses too


def test_blocked_model_does_not_start_a_run(tmp_path):
    simulate_killed_run(tmp_path)
    ran = []
    with pytest.raises(EmbeddingUnavailableError):
        with LocalRunGuard(tmp_path).run(MODEL):
            ran.append(True)
    assert ran == []


def test_block_is_per_model(tmp_path):
    simulate_killed_run(tmp_path)
    LocalRunGuard(tmp_path).check("BAAI/bge-base-en-v1.5")  # unaffected


def test_own_in_flight_run_is_not_mistaken_for_a_dead_one(tmp_path):
    guard = LocalRunGuard(tmp_path)
    with guard.run(MODEL):
        guard.check(MODEL)  # same process, marker is ours


@dataclass
class _Cfg:
    primary: str = MODEL


@pytest.fixture
def guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_GUARD_DIR", str(tmp_path))
    local_run_guard.reset_default_guard()
    yield tmp_path
    local_run_guard.reset_default_guard()


def test_lazy_local_model_skips_without_loading_when_blocked(guard_dir, monkeypatch):
    simulate_killed_run(guard_dir)
    local_run_guard.reset_default_guard()  # the restarted process builds its own guard

    def must_not_load(name):
        raise AssertionError("model must not be loaded while blocked")

    monkeypatch.setattr(gpu_entrypoint, "_load_local_model", must_not_load)

    with pytest.raises(EmbeddingUnavailableError):
        gpu_entrypoint._LazyLocalModel(_Cfg()).encode(["hello"])


def test_lazy_local_model_encodes_and_clears_its_marker(guard_dir):
    vectors = gpu_entrypoint._LazyLocalModel(_Cfg()).encode(["hello"])

    assert len(vectors) == 1
    assert list(guard_dir.iterdir()) == []
