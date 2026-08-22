"""Entrypoint: ``python -m remote_embedder`` (or the ``remote-embedder`` script)."""
from __future__ import annotations

import uvicorn

from remote_embedder.config import settings
from remote_embedder.logging_config import configure_logging


def main() -> None:
    configure_logging(settings)
    # log_config=None: skip uvicorn's own dictConfig setup so its loggers keep
    # propagating to the root logger we just configured (console + bounded
    # file), instead of getting their own separate console-only handlers.
    # access_log=False: the client hits GET /v1/models before every batch —
    # logging one line per health probe would be pure noise, not "key info".
    # Single worker on purpose: the manager owns one model in one process's
    # VRAM. Multiple uvicorn workers would each load their own copy.
    uvicorn.run(
        "remote_embedder.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,
        access_log=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
