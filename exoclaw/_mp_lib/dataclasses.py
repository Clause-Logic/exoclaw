# Minimal ``dataclasses`` stub for MicroPython.
#
# Just enough to satisfy ``from dataclasses import dataclass, field``
# usage in exoclaw core. The decorator builds an ``__init__`` from
# class-level annotations and class-level defaults, plus ``field()``
# entries — enough for the dataclass-as-record pattern that exoclaw
# uses in ``providers/types.py``, ``bus/events.py``, ``executor.py``.
#
# **What's NOT supported (intentionally):**
#
# - ``frozen=True``, ``eq``/``order`` overrides, ``__repr__`` /
#   ``__hash__`` synthesis. exoclaw's dataclasses are mutable record
#   types; equality and repr fall back to the default ``object``
#   semantics, which is fine for the cooperative-single-task model.
# - ``__post_init__``. None of the core dataclasses use it; if a
#   future one needs it, add support here.
# - Validation of ``default_factory`` vs mutable default. CPython
#   raises on ``field(default=[])``; this stub doesn't bother.
#
# Lives under ``exoclaw/_mp_lib/`` — see the README in that
# directory for how downstream firmware authors freeze it via the
# manifest. micropython-lib doesn't ship a dataclasses package, so
# there's no upstream alternative; if [micropython-lib#XYZ] ever
# adds one we'll require() it instead.


_MISSING = object()


class _Field:
    __slots__ = ("default", "default_factory")

    def __init__(self, default=_MISSING, default_factory=_MISSING):
        self.default = default
        self.default_factory = default_factory


def field(*, default=_MISSING, default_factory=_MISSING, **kwargs):
    """Mirror ``dataclasses.field``. Returns a ``_Field`` marker that
    the decorator inspects when building ``__init__``.

    Extra kwargs (``repr=``, ``compare=``, ``init=``, ``hash=``,
    ``metadata=``, ``kw_only=``) are accepted and ignored — this stub
    doesn't synthesize ``__repr__``/``__eq__``, so those flags don't
    do anything anyway. Accepting them lets exoclaw declare fields
    the same way on both runtimes."""
    return _Field(default=default, default_factory=default_factory)


def _annotations(cls):
    """Walk the inheritance chain bottom-up so subclass field
    declarations override base-class ones. MicroPython doesn't
    expose ``__mro__``, so we walk ``__bases__`` recursively.

    Needed for ``InboundMessage(Event)`` / ``OutboundMessage(Event)``
    in ``bus/events.py`` where the base class declares ``timestamp``
    and the subclass adds more fields after it."""
    chain = []

    def _walk(c):
        if c is object:
            return
        for base in getattr(c, "__bases__", ()):
            _walk(base)
        if c not in chain:
            chain.append(c)

    _walk(cls)
    seen = []
    for base in chain:
        for name in getattr(base, "__annotations__", {}):
            if name not in seen:
                seen.append(name)
    return seen


def _make_init(cls, fields):
    """Build ``__init__`` with positional + keyword args matching the
    field order. Defaults pulled from class attributes (regular
    defaults) or ``_Field`` markers (``field(default=...)`` /
    ``field(default_factory=...)``)."""

    def __init__(self, *args, **kwargs):
        # Apply positional args in declared order
        for i, name in enumerate(fields):
            if i < len(args):
                setattr(self, name, args[i])
            elif name in kwargs:
                setattr(self, name, kwargs.pop(name))
            else:
                # Fall back to class-level default (which may be a
                # ``_Field`` marker or a plain value).
                attr = getattr(cls, name, _MISSING)
                if isinstance(attr, _Field):
                    if attr.default_factory is not _MISSING:
                        # ``default_factory`` is typed as ``object``
                        # because the slot also holds the ``_MISSING``
                        # sentinel; the ``is not _MISSING`` guard
                        # above narrows it to a real callable in
                        # practice.
                        setattr(self, name, attr.default_factory())  # type: ignore[call-non-callable]
                    elif attr.default is not _MISSING:
                        setattr(self, name, attr.default)
                    else:
                        raise TypeError("missing required field: {!r}".format(name))
                elif attr is _MISSING:
                    raise TypeError("missing required field: {!r}".format(name))
                else:
                    setattr(self, name, attr)
        if kwargs:
            raise TypeError("unexpected keyword arguments: {!r}".format(list(kwargs)))

    return __init__


def dataclass(cls=None, **kwargs):
    """No-frills ``@dataclass`` decorator.

    Builds ``__init__`` from annotations + class-level defaults.
    Ignores ``frozen``, ``eq``, ``order``, ``repr``, ``init`` kwargs
    — exoclaw's dataclasses are mutable record types where the
    default ``object.__eq__`` / ``object.__repr__`` are good enough.
    """

    def _decorate(cls):
        fields = _annotations(cls)
        cls.__init__ = _make_init(cls, fields)
        # Strip ``_Field`` markers from class-level defaults so callers
        # that read ``Foo.attr`` after instantiation don't get the
        # marker object back.
        for name in fields:
            attr = getattr(cls, name, _MISSING)
            if isinstance(attr, _Field):
                delattr(cls, name)
        return cls

    if cls is None:
        return _decorate
    return _decorate(cls)
