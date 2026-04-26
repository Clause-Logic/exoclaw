"""CPython-side wrapper: drives the MicroPython matrix half.

Runs ``tests/_micropython_runner/run.py`` under MicroPython,
captures its JSON report (per-test pass/fail + per-file line
coverage), asserts every test passed, and asserts coverage on
``exoclaw/_compat.py`` ≥ 95% of the MicroPython-reachable line set.

The "MicroPython-reachable line set" is computed by parsing the
source and excluding:

- blank lines and comments (no executable content)
- lines tagged ``# pragma: no cover (micropython)`` and the
  contiguous indented block beneath them (the CPython-only
  branches)

This mirrors what coverage.py does for CPython on its own report —
each runtime measures only what it actually reaches.

**Skipping vs failing:** if the MicroPython binary path
(``$EXOCLAW_MICROPYTHON_BIN`` or ``micropython`` on PATH) is
missing, the test SKIPS with a clear message. If the binary is
found but doesn't have ``sys.settrace`` (brew bottle does NOT —
need the ``coverage`` variant built from source), the test FAILS
loudly so the CI / local rig is forced to wire up the right binary.

Build the right binary locally::

    git clone --depth 1 https://github.com/micropython/micropython ~/dev/micropython
    cd ~/dev/micropython/ports/unix && make submodules && make VARIANT=coverage
    export EXOCLAW_MICROPYTHON_BIN=~/dev/micropython/ports/unix/build-coverage/micropython

The CI workflow does the same; see ``.github/workflows/pr.yml``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_RUNNER = _REPO_ROOT / "tests" / "_micropython_runner" / "run.py"
_MICRO_TESTS_DIR = _REPO_ROOT / "tests" / "micro"
_STUBS_DIR = _REPO_ROOT / "tests" / "_micropython_stubs"
_COMPAT_SRC = _REPO_ROOT / "exoclaw" / "_compat.py"

_COVERAGE_THRESHOLD = 0.95


def _resolve_micropython_bin() -> str | None:
    """Return the MicroPython binary to use, or ``None`` to skip.

    Honors ``$EXOCLAW_MICROPYTHON_BIN`` first (CI / explicit
    override), then ``micropython`` on PATH.
    """
    env = os.environ.get("EXOCLAW_MICROPYTHON_BIN")
    if env:
        return env
    return shutil.which("micropython")


def _has_settrace(binary: str) -> bool:
    """Verify the binary supports ``sys.settrace`` — required for
    coverage measurement. Brew's bottle does NOT; the ``coverage``
    variant built from source does.
    """
    result = subprocess.run(
        [binary, "-c", "import sys; print(hasattr(sys, 'settrace'))"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() == "True"


# ── Source-level reachability analysis ───────────────────────────────────────


def _executable_lines_for(runtime: str, source_path: Path) -> set[int]:
    """Return the set of line numbers in ``source_path`` that are
    executable on the given runtime (``"cpython"`` or
    ``"micropython"``).

    Delegates to coverage.py's ``PythonParser`` for the canonical
    executable-line set (it knows about docstrings, multi-line
    tokens, decorators, etc. — all the things ``sys.settrace``
    actually reports). Per-runtime exclusion is applied by passing
    a regex that matches the OPPOSITE runtime's pragma:

    - For ``runtime == "micropython"``: exclude lines tagged
      ``# pragma: no cover (cpython)`` (CPython-only branches —
      not reachable on micro). The bare ``# pragma: no cover``
      is also excluded, matching CPython's default.
    - For ``runtime == "cpython"``: exclude lines tagged
      ``# pragma: no cover (micropython)``. Mirror policy.

    This means each runtime measures only the lines it can actually
    reach. Source-level annotation determines the scope; runtime
    tracers determine the hits.
    """
    from coverage.python import PythonParser

    # Pragma semantics: ``# pragma: no cover (X)`` means "exclude
    # this line from runtime X's coverage report" — i.e. the line
    # is unreachable on X. So for the MicroPython runtime we
    # exclude lines tagged ``(micropython)`` (the CPython-only
    # branches like ``else:`` after ``if IS_MICROPYTHON:``); for
    # the CPython runtime we exclude ``(cpython)``-tagged lines
    # (the MicroPython-only branches). Bare ``# pragma: no cover``
    # (no tag) excludes from both — coverage.py's default
    # behaviour, kept here for fallback paths like a missing
    # optional dep that neither runtime can normally reach.
    bare = r"pragma:\s*no\s*cover\s*(?:#|$)"
    if runtime == "micropython":
        tagged = r"pragma:\s*no\s*cover\s*\(micropython\)"
    elif runtime == "cpython":
        tagged = r"pragma:\s*no\s*cover\s*\(cpython\)"
    else:
        raise ValueError(f"unknown runtime: {runtime!r}")
    exclude = f"({bare}|{tagged})"

    parser = PythonParser(filename=str(source_path), exclude=exclude)
    parser.parse_source()

    # ``parser.excluded`` is line-level only. We also want to drop
    # the body of any block whose header line matched the pragma
    # (e.g. ``else:  # pragma: no cover (cpython)`` should also
    # exclude every statement inside the else clause). Walk
    # forward by indentation: any line below the matched header
    # whose indent is strictly greater is part of the block body.
    lines = source_path.read_text().splitlines()
    expanded_excluded = set(parser.excluded)

    def _indent(s: str) -> int:
        return len(s) - len(s.lstrip())

    for header_lineno in sorted(parser.excluded):
        if header_lineno > len(lines):
            continue
        header_line = lines[header_lineno - 1]
        if not header_line.rstrip().endswith(":"):
            continue
        header_indent = _indent(header_line)
        i = header_lineno + 1
        while i <= len(lines):
            line = lines[i - 1]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if _indent(line) <= header_indent:
                break
            expanded_excluded.add(i)
            i += 1

    return parser.statements - expanded_excluded


# ── Subprocess driver ────────────────────────────────────────────────────────


def _run_micropython_suite(binary: str, tmp_path: Path) -> dict:
    """Subprocess-run the MicroPython runner. Returns the parsed
    JSON report. Raises on non-zero exit (the runner prints the
    report regardless of test pass/fail, but exit 1 means at
    least one test failed; we return the report anyway so the
    caller can surface specific failures).
    """
    # Stage a flat layout so the runner's ``import _compat`` finds
    # the shim without going through ``exoclaw/__init__.py``
    # (which imports unmigrated CPython-only modules).
    stage = tmp_path / "stage"
    stage.mkdir()
    shutil.copy(_COMPAT_SRC, stage / "_compat.py")
    for stub in _STUBS_DIR.glob("*.py"):
        shutil.copy(stub, stage / stub.name)

    env = os.environ.copy()
    # Path to the staged shim + tests dir + runner imports.
    env["MICROPYPATH"] = ":{stage}".format(stage=stage)

    result = subprocess.run(
        [
            binary,
            str(_RUNNER),
            "--tests-dir",
            str(_MICRO_TESTS_DIR),
            "--cov",
            str(stage / "_compat.py"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(_REPO_ROOT),
    )

    # Last non-empty stdout line is the JSON report; earlier lines
    # are stub-logger output from the tests under cov.
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not stdout_lines:
        raise AssertionError(
            "MicroPython runner produced no output. stdout={!r} stderr={!r}".format(
                result.stdout, result.stderr
            )
        )
    try:
        report = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as e:
        raise AssertionError(
            "MicroPython runner output not JSON-parseable. last_line={!r} stderr={!r}".format(
                stdout_lines[-1], result.stderr
            )
        ) from e

    report["_returncode"] = result.returncode
    report["_staged_compat"] = str(stage / "_compat.py")
    return report


# ── Tests ────────────────────────────────────────────────────────────────────


def test_micropython_suite_passes_with_coverage(tmp_path: Path) -> None:
    """Run the MicroPython half of the matrix:

    1. Subprocess-execute the runner under MicroPython.
    2. Assert every discovered test passed.
    3. Assert ``exoclaw/_compat.py`` coverage ≥ 95% of the
       MicroPython-reachable line set.

    This is the ONE test on the CPython side that proves the
    MicroPython port works. The actual assertions live in
    ``tests/micro/test_compat.py`` and run inside MicroPython.
    """
    binary = _resolve_micropython_bin()
    if binary is None:
        pytest.skip(
            "no MicroPython binary found. Set $EXOCLAW_MICROPYTHON_BIN or "
            "install via brew. For coverage support, build the unix port's "
            "``coverage`` variant from source — see this file's docstring."
        )

    if not _has_settrace(binary):
        pytest.skip(
            "MicroPython binary at {!r} does not support sys.settrace "
            "(required for coverage measurement). Build the unix port's "
            "``coverage`` variant — brew's bottle doesn't enable it. "
            "Set $EXOCLAW_MICROPYTHON_BIN to point at the coverage build. "
            "See this file's docstring for instructions.".format(binary)
        )

    report = _run_micropython_suite(binary, tmp_path)

    # ── Pass/fail assertion ─────────────────────────────────────
    failed = report.get("failed", [])
    assert not failed, "MicroPython tests failed:\n  " + "\n  ".join(
        "{file}::{test} — {error}".format(**f) for f in failed
    )
    passed = report.get("passed", [])
    assert passed, "MicroPython runner reported no tests — discovery may be broken"

    # ── Coverage assertion ──────────────────────────────────────
    # Runner reports covered lines keyed by the path it traced —
    # which is the staged copy under ``tmp_path``. Compare against
    # the executable line set computed from the source file.
    staged_path = report["_staged_compat"]
    covered = set(report["covered"].get(staged_path, []))
    executable = _executable_lines_for("micropython", _COMPAT_SRC)

    if not executable:
        pytest.fail(
            "Could not determine MicroPython-executable line set for "
            "{}. Source-level analysis broke.".format(_COMPAT_SRC)
        )

    hit = covered & executable
    miss = executable - covered
    coverage = len(hit) / len(executable)

    assert coverage >= _COVERAGE_THRESHOLD, (
        "MicroPython coverage of exoclaw/_compat.py is "
        "{:.1%} ({}/{}); threshold is {:.0%}. "
        "Uncovered lines: {}".format(
            coverage,
            len(hit),
            len(executable),
            _COVERAGE_THRESHOLD,
            sorted(miss)[:20],
        )
    )
