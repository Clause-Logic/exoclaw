"""Event types for the message bus.

CPython gets real ``@dataclass`` so downstream callers (DBOS
journaling, openclaw's session JSONL serializer, anything using
``dataclasses.asdict``) keep working unchanged. MicroPython gets
the same classes as plain classes with hand-written ``__init__``
because MP strips ``name: type`` annotations at compile time, so
a runtime ``@dataclass`` decorator can't introspect them.

Same constructor signatures + same attribute shape on both
runtimes — only difference is ``@dataclass`` machinery (asdict,
__eq__, __repr__) is CPython-only.
"""

from datetime import datetime

from exoclaw._compat import IS_MICROPYTHON

if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass, field

    @dataclass
    class InboundMessage:
        """Message received from a chat channel."""

        channel: str  # telegram, discord, slack, whatsapp
        sender_id: str  # User identifier
        chat_id: str  # Chat/channel identifier
        content: str  # Message text
        timestamp: datetime = field(default_factory=datetime.now)
        media: list[str] = field(default_factory=list)
        metadata: dict[str, object] = field(default_factory=dict)
        session_key_override: str | None = None
        model_override: str | None = None

        @property
        def session_key(self) -> str:
            """Unique key for session identification."""
            return self.session_key_override or f"{self.channel}:{self.chat_id}"

    @dataclass
    class OutboundMessage:
        """Message to send to a chat channel."""

        channel: str
        chat_id: str
        content: str
        reply_to: str | None = None
        media: list[str] = field(default_factory=list)
        metadata: dict[str, object] = field(default_factory=dict)

else:  # pragma: no cover (cpython)

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
            self.channel = channel
            self.sender_id = sender_id
            self.chat_id = chat_id
            self.content = content
            # ``datetime.now()`` resolved at call time, not import
            # time — mirrors ``field(default_factory=datetime.now)``
            # so each new message gets a fresh stamp.
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
