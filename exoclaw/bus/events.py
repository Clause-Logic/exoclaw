"""Event types for the message bus.

Plain classes with explicit ``__init__`` rather than ``@dataclass`` —
MicroPython 1.27 strips ``name: type`` annotations at compile time, so
a runtime dataclass decorator can't introspect them. Manually-written
``__init__`` is the cross-runtime path; CPython callers see the same
keyword arguments + defaults they'd get from ``@dataclass``.
"""

from datetime import datetime


class InboundMessage:
    """Message received from a chat channel."""

    def __init__(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        timestamp: datetime | None = None,
        media: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        session_key_override: str | None = None,
        model_override: str | None = None,
    ) -> None:
        self.channel = channel  # telegram, discord, slack, whatsapp
        self.sender_id = sender_id  # User identifier
        self.chat_id = chat_id  # Chat/channel identifier
        self.content = content  # Message text
        # ``datetime.now()`` resolved at call time, not import time —
        # mirrors ``field(default_factory=datetime.now)`` semantics so
        # each new message gets a fresh stamp.
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        self.media = media if media is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.session_key_override = session_key_override
        self.model_override = model_override

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


class OutboundMessage:
    """Message to send to a chat channel."""

    def __init__(
        self,
        channel: str,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        media: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.channel = channel
        self.chat_id = chat_id
        self.content = content
        self.reply_to = reply_to
        self.media = media if media is not None else []
        self.metadata = metadata if metadata is not None else {}
