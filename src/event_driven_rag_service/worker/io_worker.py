"""
IO worker — handles IO-intensive tasks such as remote API inference calls.

Not resource-constrained by CPU or GPU, so multiple processes can be spun up
to handle higher throughput (e.g. parallel ChatGPT API calls).

Status: NOT IMPLEMENTED — stub only.  Extend when the inference pipeline is
built.  The worker inherits BaseWorker so the RabbitMQ mechanics are ready;
only ``process()`` needs an implementation.
"""
from __future__ import annotations

import logging

from event_driven_rag_service.worker.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class IoWorker(BaseWorker):
    """
    Stub IO worker.

    Queue   : ``io.infer_api.*``  (not yet declared)
    Handles : remote LLM API calls (e.g. chatgpt-4o), long downloads, etc.

    Parameters
    ----------
    rabbitmq_url : pika-compatible RabbitMQ URL
    queue_name   : queue to consume from (set when implemented)
    """

    def __init__(self, rabbitmq_url: str, queue_name: str) -> None:
        super().__init__(rabbitmq_url, queue_name, prefetch=8)

    def process(self, payload: dict) -> None:
        raise NotImplementedError("IoWorker is not yet implemented")
