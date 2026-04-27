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
_EXOCLAW_PKG = _REPO_ROOT / "exoclaw"

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
    actually reports). Per-runtime exclusion follows the pragma
    semantics ``# pragma: no cover (X)`` = "this line is
    unreachable on runtime X" — the regex for runtime X matches
    its OWN-tagged pragma:

    - For ``runtime == "micropython"``: exclude lines tagged
      ``# pragma: no cover (micropython)`` (the CPython-only
      branches that the MP bytecode never reaches). Bare
      ``# pragma: no cover`` is also excluded.
    - For ``runtime == "cpython"``: exclude lines tagged
      ``# pragma: no cover (cpython)``. Mirror policy.

    This means each runtime measures only the lines it can
    actually reach. Source-level annotation determines the scope;
    runtime tracers determine the hits.
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
    # Protocol body ``...`` mirrors ``[tool.coverage.report]`` in
    # ``pyproject.toml`` — bare ellipsis on its own line is the
    # standard idiom for unreachable abstract method bodies.
    ellipsis = r"^\s*\.\.\.\s*$"
    parts = [bare, tagged, ellipsis]
    # ``if TYPE_CHECKING:`` blocks never execute at runtime on either
    # runtime — ``TYPE_CHECKING`` is a hard ``False`` constant. Block
    # expansion below cascades the exclusion to the body.
    parts.append(r"^\s*if TYPE_CHECKING:\s*$")
    if runtime == "micropython":
        # MicroPython's ``sys.settrace`` doesn't fire ``line`` events
        # on def / class headers whose body is a same-line ``...``
        # (the inline-stub idiom for Protocol methods, abstract
        # methods, etc.). coverage.py counts them as statements on
        # CPython where the tracer DOES fire on them, but on MP
        # they're permanently uncovered through no fault of any
        # test. Exclude them from the MP-reachable set.
        inline_stub = r":\s*\.\.\.\s*(#.*)?$"
        parts.append(inline_stub)
        # NOTE: an earlier version of this code excluded ``name: Type``
        # annotation-only lines via regex. coverage.py treats matched
        # lines as part of the surrounding logical statement, so a
        # match on a function-parameter continuation line ended up
        # excluding the entire ``def`` and its body — drastic over-
        # exclusion. Whole-class-body annotations (e.g. TypedDict
        # fields, Protocol attrs) are now handled with explicit
        # ``# pragma: no cover (micropython)`` comments at the source.
    exclude = "(" + "|".join(parts) + ")"

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
        # Strip trailing comments before checking for the suite-
        # opening ``:``. Common case: ``if IS_MICROPYTHON:  # pragma:
        # no cover (cpython)`` — without this strip the rstrip-only
        # check sees the closing ``)`` of the pragma, fails, and the
        # block body never gets expanded.
        header_code = header_line.split("#", 1)[0].rstrip()
        if not header_code.endswith(":"):
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
    """Subprocess-run the MicroPython runner and return the parsed
    JSON report.

    The runner prints the report regardless of test pass/fail, so
    this helper returns the parsed report even when the subprocess
    exits non-zero — callers inspect ``report["_returncode"]`` to
    surface specific failures with full context. Only raises if
    the runner produces no output or its final stdout line isn't
    valid JSON (i.e. the runner itself crashed before emitting
    its report).

    Stages the full ``exoclaw/`` package + vendored typing /
    dataclasses / datetime stubs into ``tmp_path`` so the runner
    imports the same source layout exoclaw ships with — no flat
    copies, no ``import _compat`` shortcuts. Coverage is traced
    against every ``.py`` under the staged ``exoclaw/`` so the MP
    matrix entry covers the same source files as CPython's
    coverage.py run.
    """
    stage = tmp_path / "stage"
    stage.mkdir()
    # Copy the full package preserving directory structure. The
    # runner sets ``MICROPYPATH`` to the stage so ``import exoclaw``
    # resolves through this copy.
    shutil.copytree(_EXOCLAW_PKG, stage / "exoclaw")
    # Vendored stubs sit at the stage root so ``import typing`` /
    # ``import dataclasses`` / ``import datetime`` resolve to the
    # stubs before MicroPython's frozen modules. ``.frozen`` stays
    # earlier on the path so frozen ``asyncio`` still wins.
    for stub in _STUBS_DIR.glob("*.py"):
        shutil.copy(stub, stage / stub.name)

    env = os.environ.copy()
    # ``.frozen`` first so MP's frozen ``asyncio`` resolves before
    # anything in the stage; stage second so vendored stubs override
    # any missing stdlib (``typing``, ``dataclasses``, ``datetime``).
    env["MICROPYPATH"] = ".frozen:{stage}".format(stage=stage)

    result = subprocess.run(
        [
            binary,
            str(_RUNNER),
            "--tests-dir",
            str(_MICRO_TESTS_DIR),
            "--cov-dir",
            str(stage / "exoclaw"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
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
    report["_stage"] = str(stage)
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
        message = (
            "MicroPython binary at {!r} does not support sys.settrace "
            "(required for coverage measurement). Build the unix port's "
            "``coverage`` variant — brew's bottle doesn't enable it. "
            "Set $EXOCLAW_MICROPYTHON_BIN to point at the coverage build. "
            "See this file's docstring for instructions."
        ).format(binary)
        # If the user explicitly wired up a binary (CI / local
        # ``mise.local.toml``), a missing-settrace bottle is a
        # configuration bug, not "tests not applicable" — fail
        # loudly so the gate doesn't silently pass on a non-coverage
        # build. Reserve ``skip`` for the genuine no-binary-in-PATH
        # case where the matrix entry isn't set up at all.
        if os.environ.get("EXOCLAW_MICROPYTHON_BIN") or os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)

    report = _run_micropython_suite(binary, tmp_path)

    # The runner emits ``{"error": ...}`` when it can't even
    # discover tests (e.g. no test files matched the pattern).
    # Surface that explicitly — the failed-tests assertion below
    # would silently pass on an empty failed list.
    if report.get("error"):
        pytest.fail("MicroPython runner reported error: {}".format(report["error"]))
    if report["_returncode"] != 0 and not report.get("failed"):
        pytest.fail(
            "MicroPython runner exited {} with no tests in failed list. "
            "Output may have been truncated. Report: {}".format(report["_returncode"], report)
        )

    # ── Pass/fail assertion ─────────────────────────────────────
    failed = report.get("failed", [])
    assert not failed, "MicroPython tests failed:\n  " + "\n  ".join(
        "{file}::{test} — {error}".format(**f) for f in failed
    )
    passed = report.get("passed", [])
    assert passed, "MicroPython runner reported no tests — discovery may be broken"

    # ── Coverage assertion ──────────────────────────────────────
    # Sum coverage across every staged source file in
    # ``exoclaw/`` — same scope CPython's coverage.py measures.
    # The runner reports per-file covered lines keyed by absolute
    # path under ``tmp_path/stage/exoclaw/...``; map each back to
    # the original source under ``exoclaw/`` to compute the
    # executable line set with ``coverage.PythonParser``.
    stage_root = Path(report["_stage"]) / "exoclaw"
    covered_by_file = report["covered"]

    total_executable: int = 0
    total_hit: int = 0
    per_file: dict[str, tuple[int, int, set[int]]] = {}
    for staged_abs, hit_lines in covered_by_file.items():
        try:
            rel = Path(staged_abs).relative_to(stage_root)
        except ValueError:
            # Tracer caught a path outside the staged exoclaw — skip
            # (test-driver code, stub modules, etc.).
            continue
        original = _EXOCLAW_PKG / rel
        if not original.exists():
            continue
        executable = _executable_lines_for("micropython", original)
        if not executable:
            continue
        hit = set(hit_lines) & executable
        miss = executable - set(hit_lines)
        total_executable += len(executable)
        total_hit += len(hit)
        per_file[str(rel)] = (len(hit), len(executable), miss)

    if total_executable == 0:
        pytest.fail(
            "MicroPython coverage report had zero traceable lines. "
            "Stage root={!r}; covered keys={!r}".format(str(stage_root), list(covered_by_file)[:5])
        )

    coverage = total_hit / total_executable
    if coverage < _COVERAGE_THRESHOLD:
        # Surface the worst-covered files so failures are
        # actionable — sorted by miss count descending, top 5.
        worst = sorted(
            ((rel, hit, total, miss) for rel, (hit, total, miss) in per_file.items()),
            key=lambda x: -len(x[3]),
        )[:5]
        details = "\n  ".join(
            "{}: {}/{} ({:.0%}) — missing {}".format(
                rel, hit, total, hit / total if total else 0.0, sorted(miss)[:10]
            )
            for rel, hit, total, miss in worst
        )
        pytest.fail(
            "MicroPython coverage across exoclaw/ is {:.1%} "
            "({}/{}); threshold is {:.0%}.\n  Worst files:\n  {}".format(
                coverage, total_hit, total_executable, _COVERAGE_THRESHOLD, details
            )
        )
