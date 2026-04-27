# `exoclaw._mp_lib` — MicroPython runtime fillers

These modules **are not imported by core's CPython code.** CPython
already ships `typing` and `dataclasses` in stdlib. This directory
exists so plugin authors building MicroPython firmware can freeze
core's MP-only library fillers into their image without rolling
their own.

## What's here

- `typing.py` — `Protocol`, `TypedDict`, `TYPE_CHECKING`,
  subscriptable annotation placeholders. Just enough to satisfy
  the `from typing import …` lines core + plugins emit. The full
  CPython `typing` module isn't recreated; with `from __future__
  import annotations` enabled everywhere, annotations evaluate as
  strings and only the runtime-touched names need to exist.
- `dataclasses.py` — minimal `@dataclass` + `field` shapes.
  MicroPython 1.27 strips `name: type` annotations at compile
  time, so the upstream `dataclasses` decorator can't introspect
  fields. Core + plugins use a dual-class pattern (`@dataclass`
  on CPython, hand-written `__init__` on MP) for hot classes;
  this stub covers the cold path where the import would otherwise
  crash before the runtime gate is reached.

## Why these and not others

micropython-lib already ships `__future__`, `datetime`, `os`,
`base64`, etc. — pull those in via `require("name")` in your
manifest. We only own the modules that micropython-lib doesn't:
right now that's `typing` and `dataclasses`.

If [micropython#15911](https://github.com/micropython/micropython/pull/15911)
ever lands, MP will ship `typing` natively; we delete `typing.py`
and `require("typing")` from the manifest.

## How to use in a firmware manifest

```python
# packages/exoclaw-firmware/manifest.py (or wherever)
require("__future__")
require("datetime")
freeze("$(MPY_DIR)/extmod/asyncio")

# Pull in core's MP-runtime fillers. ``EXOCLAW_DIR`` is set by
# the build invocation to point at a checked-out core source tree.
freeze("$(EXOCLAW_DIR)/exoclaw/_mp_lib")
```

The directory is included in the published wheel + sdist via
`tool.hatch.build`, so a future `pip install exoclaw` puts it on
disk where firmware builds can find it.
