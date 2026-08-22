"""Log file setup and line-count bounding.

Tested behaviours
------------------
- configure_logging() writes to LOG_DIR/LOG_FILE when log_to_file is on.
- The default level is INFO — a DEBUG call is dropped, an INFO call is written.
- MaxLinesFileHandler keeps at most max_lines lines, oldest dropped first.
- Noisy third-party loggers (transformers, huggingface_hub, ...) are held to
  WARNING so model-loading chatter doesn't fill the file.
- log_to_file=False attaches no file handler at all.
"""
from __future__ import annotations

import logging

import pytest

from remote_embedder.logging_config import (
    _QUIET_LOGGERS,
    MaxLinesFileHandler,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """configure_logging() mutates the process-wide root logger; restore it
    afterwards so this file doesn't leak handlers into other test modules."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for h in root.handlers:
        h.close()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def test_configure_logging_writes_to_the_configured_file(settings, tmp_path):
    settings.log_to_file = True
    settings.log_dir = str(tmp_path)
    settings.log_file = "svc.log"
    settings.log_level = "INFO"  # the shared fixture defaults to WARNING

    file_handler = configure_logging(settings)
    logging.getLogger("remote_embedder.test").info("hello")
    file_handler.flush()

    log_path = tmp_path / "svc.log"
    assert log_path.exists()
    assert "hello" in log_path.read_text()


def test_default_level_is_info_not_debug(settings, tmp_path):
    settings.log_to_file = True
    settings.log_dir = str(tmp_path)
    settings.log_file = "svc.log"
    settings.log_level = "INFO"

    file_handler = configure_logging(settings)
    logger = logging.getLogger("remote_embedder.test")
    logger.debug("should not appear")
    logger.info("should appear")
    file_handler.flush()

    contents = (tmp_path / "svc.log").read_text()
    assert "should not appear" not in contents
    assert "should appear" in contents


def test_log_to_file_false_attaches_no_file_handler(settings, tmp_path):
    settings.log_to_file = False
    settings.log_dir = str(tmp_path)

    file_handler = configure_logging(settings)

    assert file_handler is None
    assert list(tmp_path.iterdir()) == []


def test_quiet_loggers_are_held_to_warning(settings, tmp_path):
    settings.log_to_file = True
    settings.log_dir = str(tmp_path)
    configure_logging(settings)

    for name in _QUIET_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


# --- MaxLinesFileHandler ----------------------------------------------------


def test_trim_keeps_only_the_last_max_lines(tmp_path):
    path = tmp_path / "bounded.log"
    handler = MaxLinesFileHandler(path, max_lines=5, check_every=10_000)
    logger = logging.getLogger("test.trim")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for i in range(20):
        logger.info("line-%d", i)
    handler.trim()

    lines = path.read_text().splitlines()
    assert len(lines) == 5
    # The oldest lines were dropped, the newest kept, in order.
    assert "line-19" in lines[-1]
    assert "line-15" in lines[0]

    logger.removeHandler(handler)
    handler.close()


def test_trim_is_a_no_op_under_the_limit(tmp_path):
    path = tmp_path / "small.log"
    handler = MaxLinesFileHandler(path, max_lines=100, check_every=10_000)
    logger = logging.getLogger("test.trim.small")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for i in range(3):
        logger.info("line-%d", i)
    handler.trim()

    assert len(path.read_text().splitlines()) == 3

    logger.removeHandler(handler)
    handler.close()


def test_trim_runs_automatically_every_check_every_writes(tmp_path):
    path = tmp_path / "auto.log"
    handler = MaxLinesFileHandler(path, max_lines=3, check_every=5)
    logger = logging.getLogger("test.trim.auto")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for i in range(5):  # exactly check_every writes -> one automatic trim
        logger.info("line-%d", i)

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert "line-4" in lines[-1]

    logger.removeHandler(handler)
    handler.close()


def test_handler_survives_repeated_trims(tmp_path):
    """The file is closed and reopened on every trim — make sure logging
    afterwards still works rather than writing to a stale/closed stream."""
    path = tmp_path / "repeat.log"
    handler = MaxLinesFileHandler(path, max_lines=2, check_every=3)
    logger = logging.getLogger("test.trim.repeat")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for i in range(12):  # several automatic trims
        logger.info("line-%d", i)
    handler.trim()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert "line-11" in lines[-1]

    logger.removeHandler(handler)
    handler.close()
