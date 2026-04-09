from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Message(Generic[T]):
    """Typed envelope passed between publishers and subscribers."""

    topic: str
    payload: T


class Subscriber(Protocol[T]):
    """A consumer that reacts to messages for a topic."""

    def topics(self) -> frozenset[str]:
        """Topics this subscriber listens to."""

    def on_message(self, message: Message[T]) -> None:
        """Handle a single message."""


Callback = Callable[[Message[T]], None]
