"""Unit test fixtures.

All fixtures here are in-process only — no database, no containers, no network.
Tests in this directory run in milliseconds.

Key fixtures
------------
fake_bus      : FakeEventBus — records published events, yields on subscribe
fake_exchange : FakeExchange — records publish() calls with routing keys
"""
from __future__ import annotations

import pytest

from tests.utils.factories import FakeEventBus, FakeExchange


@pytest.fixture
def fake_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def fake_exchange() -> FakeExchange:
    return FakeExchange()
