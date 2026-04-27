# Minimal ``typing`` stub for MicroPython.
#
# Just enough to satisfy ``from typing import X`` for the symbols
# exoclaw core uses. exoclaw enables ``from __future__ import
# annotations`` everywhere, so type annotations evaluate as strings
# at runtime — none of the parameterised forms (``Protocol[T]``,
# ``Optional[X]``) actually subscribe these at import time.
#
# What DOES execute at runtime:
# - ``class Foo(Protocol):`` / ``class Foo(Generic):`` — Protocol
#   and Generic must be subclassable.
# - ``_T = TypeVar("_T")`` — the call must succeed and return
#   something usable.
# - ``cast(T, val)`` — must return ``val``.
# - ``runtime_checkable`` — must be a callable that returns its arg.
# - ``TYPE_CHECKING`` — must be ``False`` so guarded imports don't run.
#
# Vendored under ``tests/_micropython_stubs/`` so the test rig can
# import it via ``MICROPYPATH``. On a real MicroPython device, install
# via ``mip install typing`` (micropython-lib ships a similar shim).

TYPE_CHECKING = False


# ── Subclassable bases ───────────────────────────────────────────────────────


class Protocol:
    """No-op Protocol base. Subclasses are regular classes; instance
    checks fall through to standard attribute matching, which is
    fine because exoclaw doesn't rely on typing-machinery validation
    at runtime."""


class Generic:
    """No-op Generic base. Same logic as ``Protocol``."""


class TypedDict(dict):
    """Minimal TypedDict — runtime is a regular dict subclass.

    Class bodies that declare ``key: Type`` annotations work because
    those annotations are strings under ``from __future__ import
    annotations``. The ``total=`` kwarg on the class statement is
    accepted and ignored."""

    def __init_subclass__(cls, **kwargs):
        # Swallow any class-keyword arguments so subclass declarations
        # like ``class Foo(TypedDict, total=False)`` don't raise.
        # MicroPython 1.27 doesn't accept ``total=True`` as a named
        # parameter here either — catch all via ``**kwargs``.
        pass


# ── Annotation-only sentinels (subscriptable for safety, but with
#    ``from __future__ import annotations`` enabled callers should
#    rarely actually subscript these at import time) ──────────────


class _Subscriptable:
    def __getitem__(self, item):
        return self


Optional = _Subscriptable()
Union = _Subscriptable()
TypeGuard = _Subscriptable()
ClassVar = _Subscriptable()
Final = _Subscriptable()
Required = _Subscriptable()
NotRequired = _Subscriptable()
Annotated = _Subscriptable()
Literal = _Subscriptable()
Awaitable = _Subscriptable()
Callable = _Subscriptable()
Coroutine = _Subscriptable()
Iterable = _Subscriptable()
Iterator = _Subscriptable()
AsyncIterable = _Subscriptable()
AsyncIterator = _Subscriptable()
Mapping = _Subscriptable()
Sequence = _Subscriptable()
Type = _Subscriptable()


# ── Identity / pass-through helpers ──────────────────────────────────────────


def runtime_checkable(cls):
    return cls


def cast(typ, val):
    return val


def TypeVar(name, *args, **kwargs):
    """Return a placeholder for a type variable.

    The actual identity is irrelevant on MicroPython — annotations
    don't enforce anything at runtime — but the call must succeed
    and return something module-level assignment can hold."""
    return name


# ── Aliases for builtin types ────────────────────────────────────────────────

Any = object
List = list
Dict = dict
Tuple = tuple
Set = set
FrozenSet = frozenset
