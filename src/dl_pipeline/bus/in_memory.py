from __future__ import annotations

from collections import defaultdict

from dl_pipeline.bus.protocol import Callback, Message, Subscriber


class InMemoryBus:
    """Synchronous in-process pub/sub bus (useful for local pipelines and tests)."""

    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callback[object]]] = defaultdict(list)

    def subscribe(self, subscriber: Subscriber[object]) -> None:
        for topic in subscriber.topics():
            self._callbacks[topic].append(subscriber.on_message)

    def subscribe_callback(self, topic: str, callback: Callback[object]) -> None:
        self._callbacks[topic].append(callback)

    def publish(self, message: Message[object]) -> None:
        for callback in self._callbacks.get(message.topic, []):
            callback(message)
