"""
Synchronous RabbitMQ worker base.

All workers in this project are synchronous processes using pika (blocking).
Run multiple worker processes for concurrency — no async machinery needed.

Two consumption patterns
------------------------
1. **Single-message** (CPU, IO workers)
   Override ``process(payload)``.  The default ``_run_inner`` drives
   ``basic_consume`` and calls it once per message, acking on success and
   nacking to the DLQ on any exception.

2. **Batch / custom loop** (GPU worker)
   Override ``_run_inner(channel)`` entirely.  Use the ``_poll_queue``,
   ``_ack``, and ``_nack`` helpers provided here.

Failure contract
----------------
- Successful ``process()`` call  → message is acked.
- ``process()`` raises any error → message is nacked with requeue=False,
  routing it to the DLQ via ``x-dead-letter-exchange``.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from typing import Optional

import pika
import pika.exceptions
from pika.adapters.blocking_connection import BlockingChannel

logger = logging.getLogger(__name__)


class BaseWorker:
    """
    Parameters
    ----------
    rabbitmq_url : pika-compatible URL, e.g. ``amqp://guest:guest@localhost/``
    queue_name   : queue to consume from; leave blank when ``_run_inner``
                   polls multiple queues manually (e.g. the GPU worker).
    prefetch     : ``basic_qos`` prefetch_count — tune per worker type.
    """

    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str = "",
        prefetch: int = 1,
    ) -> None:
        self._rabbitmq_url = rabbitmq_url
        self._queue_name = queue_name
        self._prefetch = prefetch
        self._running = True

    # ------------------------------------------------------------------
    # RabbitMQ helpers
    # ------------------------------------------------------------------

    def _open_channel(self) -> BlockingChannel:
        """Open a fresh pika connection and return a channel with QoS applied."""
        params = pika.URLParameters(self._rabbitmq_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        if self._prefetch:
            channel.basic_qos(prefetch_count=self._prefetch)
        return channel

    def _poll_queue(
        self,
        channel: BlockingChannel,
        queue_name: str,
        max_msgs: int,
    ) -> list[tuple[int, bytes]]:
        """Pull up to *max_msgs* messages without blocking.

        Returns a list of ``(delivery_tag, body)`` pairs for valid messages.
        Messages with empty bodies are nacked to the DLQ immediately.
        """
        results: list[tuple[int, bytes]] = []
        while len(results) < max_msgs:
            method, _props, body = channel.basic_get(queue=queue_name, auto_ack=False)
            if method is None:
                break
            if not body:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                continue
            results.append((method.delivery_tag, body))
        return results

    def _ack(self, channel: BlockingChannel, delivery_tag: int) -> None:
        channel.basic_ack(delivery_tag=delivery_tag)

    def _nack(
        self,
        channel: BlockingChannel,
        delivery_tag: int,
        requeue: bool = False,
    ) -> None:
        channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _setup_signals(self) -> None:
        def _shutdown(signum, _frame) -> None:
            logger.info("Signal %s received — shutting down", signum)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start consuming.  Reconnects automatically on RabbitMQ connection loss."""
        self._setup_signals()
        logger.info(
            "%s starting (queue=%r, prefetch=%d)",
            self.__class__.__name__,
            self._queue_name,
            self._prefetch,
        )
        try:
            while self._running:
                channel: Optional[BlockingChannel] = None
                try:
                    channel = self._open_channel()
                    self._run_inner(channel)
                except KeyboardInterrupt:
                    logger.info("%s interrupted", self.__class__.__name__)
                    self._running = False
                except pika.exceptions.AMQPError as exc:
                    logger.error(
                        "RabbitMQ connection lost: %s — reconnecting in 5s", exc
                    )
                    time.sleep(5.0)
                finally:
                    if channel is not None:
                        try:
                            channel.connection.close()
                        except Exception:
                            pass
        except Exception:
            logger.exception(
                "%s: unexpected error in run loop", self.__class__.__name__
            )
        logger.info("%s shut down", self.__class__.__name__)

    def _run_inner(self, channel: BlockingChannel) -> None:
        """Default loop: ``basic_consume`` driving ``process()`` per message.

        Override this entirely for workers that need batch collection or custom
        polling logic (e.g. the GPU worker).
        """
        channel.basic_consume(
            queue=self._queue_name,
            on_message_callback=self._on_message,
            auto_ack=False,
        )
        channel.start_consuming()

    def _on_message(
        self,
        channel: BlockingChannel,
        method,
        _properties,
        body: bytes,
    ) -> None:
        """Deserialize body, call ``process()``, ack on success, nack to DLQ on failure."""
        try:
            payload = json.loads(body)
            self.process(payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception(
                "%s: error processing message (tag=%s) — routing to DLQ",
                self.__class__.__name__,
                method.delivery_tag,
            )
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def process(self, payload: dict) -> None:
        """Handle one deserialized task payload.

        Override in subclasses that use the default single-message consume loop.
        Workers that override ``_run_inner()`` do not call this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must override process() or _run_inner()"
        )
