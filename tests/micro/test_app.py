"""``Exoclaw`` composition root on MicroPython.

The ``Exoclaw`` class wires provider, conversation, channels, tools,
and the agent loop together. ``_build`` instantiates the internals;
``run`` starts everything until interrupted. Tests cover the
construction + ``_build`` paths — running indefinitely is its own
contract that doesn't fit a unit test.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""


from exoclaw.agent.loop import AgentLoop
from exoclaw.app import Exoclaw
from exoclaw.bus.queue import MessageBus
from exoclaw.channels.manager import ChannelManager
from exoclaw.providers.types import LLMResponse


class _StubProvider:
    def get_default_model(self):
        return "default-model"

    async def chat(self, messages, tools=None, model=None, **kw):
        return LLMResponse(content="ok")


class _StubConversation:
    async def build_prompt(self, session_id, message, **kw):
        return [{"role": "user", "content": message}]

    async def record(self, session_id, msgs):
        pass

    async def clear(self, session_id):
        return True

    def list_sessions(self):
        return []


def test_exoclaw_construction_minimal():
    """Required kwargs only: provider + conversation. Channels and
    tools default to empty lists."""
    app = Exoclaw(provider=_StubProvider(), conversation=_StubConversation())
    assert app.provider is not None
    assert app.conversation is not None
    assert app.channels == []
    assert app.tools == []
    assert app.bus is None
    assert app.model is None  # resolved at ``_build`` time from provider


def test_exoclaw_construction_full():
    """All kwargs populated. Verifies the constructor doesn't drop
    any of them."""
    app = Exoclaw(
        provider=_StubProvider(),
        conversation=_StubConversation(),
        channels=[],
        tools=[],
        model="custom-model",
        temperature=0.5,
        max_tokens=2048,
        max_iterations=20,
        reasoning_effort="high",
    )
    assert app.model == "custom-model"
    assert app.temperature == 0.5
    assert app.max_tokens == 2048
    assert app.max_iterations == 20
    assert app.reasoning_effort == "high"


def test_build_creates_bus_when_not_provided():
    """``_build`` defaults to a fresh ``MessageBus`` if none was
    passed at construction. Same path the canonical-usage example
    in the module docstring takes."""
    app = Exoclaw(provider=_StubProvider(), conversation=_StubConversation())
    bus, agent, mgr = app._build()
    assert isinstance(bus, MessageBus)
    assert isinstance(agent, AgentLoop)
    assert isinstance(mgr, ChannelManager)


def test_build_uses_provided_bus():
    """When a bus is passed at construction, ``_build`` uses it
    instead of creating a fresh one."""
    custom_bus = MessageBus()
    app = Exoclaw(
        provider=_StubProvider(),
        conversation=_StubConversation(),
        bus=custom_bus,
    )
    bus, _agent, _mgr = app._build()
    assert bus is custom_bus


def test_build_resolves_default_model_from_provider():
    """When ``model`` isn't provided, ``_build`` calls
    ``provider.get_default_model()`` to get one."""
    app = Exoclaw(provider=_StubProvider(), conversation=_StubConversation())
    _bus, agent, _mgr = app._build()
    assert agent.model == "default-model"


def test_build_threads_explicit_model():
    """When ``model`` is set, ``_build`` uses it (not the provider's
    default)."""
    app = Exoclaw(
        provider=_StubProvider(),
        conversation=_StubConversation(),
        model="explicit-model",
    )
    _bus, agent, _mgr = app._build()
    assert agent.model == "explicit-model"


def test_build_passes_tools_through():
    """Tools list reaches the AgentLoop's tool registry."""

    class _Echo:
        name = "echo"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kw):
            return ""

    app = Exoclaw(
        provider=_StubProvider(),
        conversation=_StubConversation(),
        tools=[_Echo()],
    )
    _bus, agent, _mgr = app._build()
    assert "echo" in agent.tools.tool_names
