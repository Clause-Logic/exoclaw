# Minimal ``datetime`` stub for MicroPython.
#
# exoclaw uses ``datetime.now()`` as a ``field(default_factory=...)``
# in ``bus/events.py`` to stamp inbound / outbound messages. The
# stamp goes into log lines and is occasionally compared for ordering
# but never serialised back to a wall-clock time on micro — so a
# wall-clock-equivalent monotonic-ish stamp is enough.
#
# MicroPython 1.27 ships ``time.time()`` and ``time.localtime()``, but
# not the full ``datetime`` module. This stub wraps them into the
# minimum shape exoclaw uses.

import time as _time


class datetime:
    """Minimal ``datetime`` replacement.

    Holds a Unix epoch seconds value; ``isoformat()`` produces a
    sortable string. Comparison delegates to the underlying float so
    ordering works without implementing full datetime arithmetic."""

    __slots__ = ("_ts",)

    def __init__(self, ts):
        self._ts = ts

    @classmethod
    def now(cls):
        return cls(_time.time())

    def isoformat(self):
        # ``time.localtime`` returns a 9-tuple with calendar fields.
        # Format like ``2026-04-26T13:34:05`` to match CPython's
        # ``isoformat()`` for the common no-microseconds case.
        t = _time.localtime(int(self._ts))
        return "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}".format(
            t[0], t[1], t[2], t[3], t[4], t[5]
        )

    def __repr__(self):
        return "datetime({})".format(self.isoformat())

    def __eq__(self, other):
        return isinstance(other, datetime) and self._ts == other._ts

    def __lt__(self, other):
        return isinstance(other, datetime) and self._ts < other._ts

    def __hash__(self):
        return hash(self._ts)
