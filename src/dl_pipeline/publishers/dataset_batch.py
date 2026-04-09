from __future__ import annotations

from collections.abc import Iterable, Iterator

from dl_pipeline.bus.in_memory import InMemoryBus
from dl_pipeline.bus.protocol import Message


class BatchPublisher:
    """Publishes fixed-size batches from a dataset iterator onto a topic."""

    def __init__(
        self,
        *,
        bus: InMemoryBus,
        topic_batches: str,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._bus = bus
        self._topic_batches = topic_batches
        self._batch_size = batch_size

    def publish_from_iterable(self, rows: Iterable[object]) -> int:
        batch: list[object] = []
        num_messages = 0
        for row in rows:
            batch.append(row)
            if len(batch) >= self._batch_size:
                self._bus.publish(Message(topic=self._topic_batches, payload=tuple(batch)))
                num_messages += 1
                batch = []
        if batch:
            self._bus.publish(Message(topic=self._topic_batches, payload=tuple(batch)))
            num_messages += 1
        return num_messages

    def stream_batches(self, rows: Iterator[object]) -> Iterator[tuple[object, ...]]:
        batch: list[object] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= self._batch_size:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)
