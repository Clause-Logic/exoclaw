# Minimal typing stub for MicroPython.
# Just enough to satisfy `from typing import X` for the symbols exoclaw uses.
TYPE_CHECKING = False


def _identity(x):
    return x


def _factory(*a, **k):
    return _identity


Any = object
Optional = _factory
Callable = _factory
List = list
Dict = dict
Tuple = tuple
Set = set


def Protocol(*a, **k):
    return type


def runtime_checkable(cls):
    return cls


def TypeGuard(*a, **k):
    return bool


class _GenericAlias:
    def __getitem__(self, _):
        return self


def _gen():
    return _GenericAlias()
