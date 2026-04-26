"""Tiny pytest-shim + line tracer for MicroPython.

Runs as the entry point of the MicroPython side of the matrix.
Discovers test files, runs every ``test_*`` function, traces line
coverage on a target file via ``sys.settrace``, and emits a JSON
report on stdout.

This file must run on MicroPython's unix port built with
``MICROPY_PY_SYS_SETTRACE=1`` (the ``coverage`` variant). The build
recipe is in ``tests/_micropython_runner/build_micropython.sh``;
locally the brew bottle does NOT enable settrace, so callers must
build the coverage variant from source and pass its path.

Usage (under MicroPython)::

    micropython run.py --tests-dir tests/micro --cov exoclaw/_compat.py

Output (one JSON object on stdout when complete)::

    {"passed": [...], "failed": [...], "covered": [int, ...]}

Exits 0 on every test passing, 1 on any failure / error / import
problem. The CPython side (``tests/test_micropython_runner.py``)
parses the JSON, asserts every test passed, computes coverage % on
the target file (counting only lines reachable from the
MicroPython runtime — see ``_executable_lines``), and asserts
``coverage >= 95%``.

Pragma protocol — runtime-specific exclusions:

- ``# pragma: no cover (cpython)``  — excluded from CPython
  coverage.py reporting (configured in ``pyproject.toml``).
- ``# pragma: no cover (micropython)``  — excluded from this
  runner's report (filtered by ``_executable_lines``).

The ``_compat.py`` shim uses both — every ``if IS_MICROPYTHON:``
branch's body is unreachable on CPython, every ``else:`` block is
unreachable on MicroPython. Each runtime measures only what it
actually reaches.
"""

import json
import os
import sys

# ── Argument parsing (no argparse on MicroPython core) ───────────────────────


def _parse_args(argv):
    """``--cov FILE`` adds a single source file to trace (repeatable);
    ``--cov-dir DIR`` adds every ``.py`` under ``DIR`` recursively.
    Both options can be combined and repeated."""
    args = {"tests_dir": "tests/micro", "cov_paths": [], "cov_dirs": []}
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--tests-dir" and i + 1 < len(argv):
            args["tests_dir"] = argv[i + 1]
            i += 2
        elif arg == "--cov" and i + 1 < len(argv):
            args["cov_paths"].append(argv[i + 1])
            i += 2
        elif arg == "--cov-dir" and i + 1 < len(argv):
            args["cov_dirs"].append(argv[i + 1])
            i += 2
        elif arg == "--help" or arg == "-h":
            print(__doc__ or "")
            sys.exit(0)
        else:
            print("unrecognised arg:", arg)
            sys.exit(2)
    return args


def _walk_py(root):
    """Recursive ``.py`` walk. MicroPython's ``os`` doesn't ship
    ``os.walk``; reimplement with ``os.listdir`` + ``os.stat`` so we
    can resolve every source file under ``exoclaw/`` for tracing.

    ``__pycache__`` and dot-prefixed directories are skipped — they're
    never user source."""
    out = []
    try:
        entries = os.listdir(root)
    except OSError:
        return out
    for name in entries:
        if name.startswith(".") or name == "__pycache__":
            continue
        full = root + "/" + name
        try:
            mode = os.stat(full)[0]
        except OSError:
            continue
        # ``stat[0]`` mode_t — bit 0o040000 indicates directory on
        # both CPython and MicroPython unix port. Standard POSIX bits.
        if mode & 0o040000:
            out.extend(_walk_py(full))
        elif name.endswith(".py"):
            out.append(full)
    return out


# ── Coverage tracer ──────────────────────────────────────────────────────────


def _make_tracer(target_paths):
    """Return a ``sys.settrace``-shaped callback that records line
    numbers hit, scoped to ``target_paths``.

    ``target_paths`` is a set of absolute paths. A frame's
    ``__file__`` (resolved to absolute) is matched against the set;
    frames outside it (test code, runtime, etc.) are ignored so the
    coverage report only covers the source under test.
    """
    covered = {p: set() for p in target_paths}

    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        # Frame's ``__file__`` is the only handle we have. Some
        # MicroPython frames may not have ``__file__`` (REPL,
        # builtin); skip those.
        path = frame.f_globals.get("__file__")
        if path is None:
            return tracer
        # ``__file__`` may be relative; canonicalise to absolute.
        abs_path = path if path.startswith("/") else os.getcwd() + "/" + path
        if abs_path in covered:
            covered[abs_path].add(frame.f_lineno)
        return tracer

    return tracer, covered


# ── Test discovery + runner ──────────────────────────────────────────────────


def _list_test_files(tests_dir):
    """Return absolute paths to every ``test_*.py`` in ``tests_dir``.

    Non-recursive — micro's test set is small and flat. Sorted by
    name so the report is stable across runs.
    """
    out = []
    abs_dir = tests_dir if tests_dir.startswith("/") else os.getcwd() + "/" + tests_dir
    for name in sorted(os.listdir(abs_dir)):
        if name.startswith("test_") and name.endswith(".py"):
            out.append(abs_dir + "/" + name)
    return out


def _import_file(path):
    """Import a test file as a fresh module.

    MicroPython's ``importlib`` is minimal — no
    ``importlib.util.spec_from_file_location``. Easiest portable
    path is to ``exec`` the source into a fresh ``dict`` and treat
    that as the module namespace.
    """
    with open(path) as f:
        src = f.read()
    # Module dict gets ``__file__`` so the tracer can scope
    # correctly when the test code itself has no nested calls into
    # the target source — and so test code that references its own
    # path works.
    ns = {"__name__": path, "__file__": path}
    exec(src, ns)  # noqa: S102 -- dynamic import is the whole point
    return ns


def _discover_test_functions(ns):
    """Return ``[(name, fn)]`` for every callable named ``test_*``
    in the module namespace, sorted by name."""
    out = []
    for name in sorted(ns):
        if name.startswith("test_") and callable(ns[name]):
            out.append((name, ns[name]))
    return out


def _run_one(test_name, fn):
    """Invoke a single test function. Returns ``(ok, error_repr)``."""
    try:
        fn()
        return True, None
    except Exception as e:
        return False, "{}: {}".format(type(e).__name__, e)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = _parse_args(sys.argv)

    cov_paths = set()
    for raw in args["cov_paths"]:
        abs_path = raw if raw.startswith("/") else os.getcwd() + "/" + raw
        cov_paths.add(abs_path)
    for raw in args["cov_dirs"]:
        abs_dir = raw if raw.startswith("/") else os.getcwd() + "/" + raw
        for f in _walk_py(abs_dir):
            cov_paths.add(f)

    test_files = _list_test_files(args["tests_dir"])
    if not test_files:
        print(json.dumps({"error": "no test files found in " + args["tests_dir"]}))
        sys.exit(1)

    tracer, covered = _make_tracer(cov_paths) if cov_paths else (None, {})

    # Activate tracing BEFORE the import phase. Module-level
    # statements in the target file (class definitions, function
    # ``def`` headers, top-level assignments) execute exactly once
    # at import; if settrace isn't active then, those lines never
    # appear in the line-event stream and they show up as
    # uncovered even though they did run.
    if tracer is not None:
        sys.settrace(tracer)

    passed = []
    failed = []
    try:
        for path in test_files:
            try:
                ns = _import_file(path)
            except Exception as e:
                failed.append(
                    {
                        "file": path,
                        "test": "<import>",
                        "error": "{}: {}".format(type(e).__name__, e),
                    }
                )
                continue

            functions = _discover_test_functions(ns)
            for name, fn in functions:
                full_name = path + "::" + name
                ok, err = _run_one(name, fn)
                if ok:
                    passed.append(full_name)
                else:
                    failed.append({"file": path, "test": name, "error": err})
    finally:
        if tracer is not None:
            sys.settrace(None)

    report = {
        "passed": passed,
        "failed": failed,
        "covered": {p: sorted(lines) for p, lines in covered.items()},
    }
    print(json.dumps(report))
    sys.exit(0 if not failed else 1)


main()
