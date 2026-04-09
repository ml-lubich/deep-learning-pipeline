from __future__ import annotations

from dl_pipeline.bus.protocol import Message


class BatchCollector:
    """Collects batch messages for assertions and lightweight logging sinks."""

    def __init__(self, topic: str) -> None:
        self._topic = topic
        self.received: list[tuple[object, ...]] = []

    def topics(self) -> frozenset[str]:
        return frozenset({self._topic})

    def on_message(self, message: Message[object]) -> None:
        payload = message.payload
        if not isinstance(payload, tuple):
            raise TypeError(f"expected tuple batch payload, got {type(payload).__name__}")
        self.received.append(payload)
