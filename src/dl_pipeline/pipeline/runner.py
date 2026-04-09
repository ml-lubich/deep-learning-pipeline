from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dl_pipeline.bus.in_memory import InMemoryBus
from dl_pipeline.publishers.dataset_batch import BatchPublisher
from dl_pipeline.subscribers.batch_collector import BatchCollector


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    num_batches_published: int
    batches_received: tuple[tuple[object, ...], ...]


def run_batch_pipeline(
    *,
    rows: Iterable[object],
    batch_size: int,
    topic_batches: str = "datasets/batches",
) -> PipelineRunResult:
    """Run a minimal publisher → bus → subscriber pipeline over tabular-style rows."""

    bus: InMemoryBus = InMemoryBus()
    collector = BatchCollector(topic=topic_batches)
    bus.subscribe(collector)
    publisher = BatchPublisher(bus=bus, topic_batches=topic_batches, batch_size=batch_size)
    num = publisher.publish_from_iterable(rows)
    return PipelineRunResult(
        num_batches_published=num,
        batches_received=tuple(collector.received),
    )
