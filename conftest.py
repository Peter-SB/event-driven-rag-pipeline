"""Root conftest.py — shared across unit, integration, and e2e tests.

Only truly global configuration lives here.  Test-level fixtures that need
infrastructure (Postgres, RabbitMQ) are in tests/integration/conftest.py and
tests/e2e/conftest.py respectively.

This file ensures:
- The src/ package is importable as event_driven_rag_service (not src.event_driven_rag_service)
- The tests/ package is importable for shared factories and utilities
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the import path when tests are run from the project root
# without the package being installed.  When installed via `pip install -e .`
# this is a no-op.
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
