"""Shared test fixtures.

Only fixtures used across *multiple* tiers (unit, integration, e2e) belong here.
Tier-specific infrastructure (testcontainers, live-stack connections) lives in
the tier's own conftest so it cannot be accidentally requested by the wrong tier:

  tests/unit/conftest.py        — in-process fakes only
  tests/integration/conftest.py — testcontainers (Postgres, RabbitMQ)
  tests/e2e/conftest.py         — connections to the live Docker Compose stack
"""
