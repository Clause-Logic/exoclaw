"""ToolBase + ToolContext on MicroPython.

``ToolBase`` is the optional mixin that ships ``cast_params`` /
``validate_params`` / ``_cast_value`` / ``_validate`` so plugin
authors don't reimplement JSON-schema enforcement. These tests
cover the schema branches the agent loop relies on every turn —
type casts, missing-required detection, enum, min/max, nested
objects, arrays.

Pure-Python — runs under ``tests/_micropython_runner/run.py``.
"""

from exoclaw.agent.tools.protocol import ToolBase, ToolContext


def test_tool_context_constructor():
    """``ToolContext`` is a plain class on both runtimes (was
    ``@dataclass`` before — annotations don't survive MP compile)."""
    ctx = ToolContext(session_key="s:1", channel="telegram", chat_id="c1")
    assert ctx.session_key == "s:1"
    assert ctx.channel == "telegram"
    assert ctx.chat_id == "c1"
    assert ctx.executor is None


def test_tool_context_with_executor():
    """``executor`` is the optional handle for durable I/O. Not
    introspected here — just verify the field round-trips."""
    sentinel = object()
    ctx = ToolContext(session_key="s", channel="c", chat_id="i", executor=sentinel)
    assert ctx.executor is sentinel


# ── ToolBase: cast_params ─────────────────────────────────────────


class _IntTool(ToolBase):
    """Tool with a single integer ``n`` parameter — covers the
    string-to-int cast branch."""

    name = "inttool"
    description = "takes an int"
    parameters = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }


def test_cast_params_string_to_int():
    """JSON schema ``integer`` + a string value → int cast."""
    tool = _IntTool()
    cast_result = tool.cast_params({"n": "42"})
    assert cast_result == {"n": 42}


def test_cast_params_string_to_number():
    """JSON schema ``number`` + a string value → float cast."""

    class _NumTool(ToolBase):
        name = "numtool"
        description = "x"
        parameters = {"type": "object", "properties": {"x": {"type": "number"}}}

    cast_result = _NumTool().cast_params({"x": "3.14"})
    assert cast_result == {"x": 3.14}


def test_cast_params_int_already_correct_type():
    """Integer value already int → no change."""
    cast_result = _IntTool().cast_params({"n": 42})
    assert cast_result == {"n": 42}


def test_cast_params_string_to_bool():
    """JSON schema ``boolean`` + a string value → True / False / passthrough."""

    class _BoolTool(ToolBase):
        name = "btool"
        description = "x"
        parameters = {"type": "object", "properties": {"flag": {"type": "boolean"}}}

    t = _BoolTool()
    assert t.cast_params({"flag": "true"}) == {"flag": True}
    assert t.cast_params({"flag": "false"}) == {"flag": False}
    assert t.cast_params({"flag": "yes"}) == {"flag": True}
    # Unknown string passes through unchanged.
    assert t.cast_params({"flag": "maybe"}) == {"flag": "maybe"}


def test_cast_params_to_string():
    """JSON schema ``string`` + a non-string value → ``str(value)``."""

    class _StrTool(ToolBase):
        name = "stool"
        description = "x"
        parameters = {"type": "object", "properties": {"s": {"type": "string"}}}

    assert _StrTool().cast_params({"s": 42}) == {"s": "42"}


def test_cast_params_passes_through_unknown_keys():
    """Keys not in the schema's properties are kept as-is."""
    cast_result = _IntTool().cast_params({"n": "1", "extra": "hello"})
    assert cast_result == {"n": 1, "extra": "hello"}


def test_cast_params_array_items():
    """``items`` schema applied per element when type=array."""

    class _ArrTool(ToolBase):
        name = "atool"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
        }

    cast_result = _ArrTool().cast_params({"ids": ["1", "2", "3"]})
    assert cast_result == {"ids": [1, 2, 3]}


# ── ToolBase: validate_params ─────────────────────────────────────


def test_validate_params_ok():
    """Well-formed params → empty error list."""
    errors = _IntTool().validate_params({"n": 42})
    assert errors == []


def test_validate_params_missing_required():
    """Required field absent → error list non-empty."""
    errors = _IntTool().validate_params({})
    assert len(errors) == 1
    assert "missing" in errors[0].lower()


def test_validate_params_wrong_type():
    """Integer field given a string → error list mentions integer."""
    errors = _IntTool().validate_params({"n": "not-a-number"})
    assert any("integer" in e.lower() for e in errors)


def test_validate_params_non_dict():
    """Top-level params must be a dict — anything else short-circuits
    with a single error string."""
    errors = _IntTool().validate_params([1, 2, 3])
    assert len(errors) == 1
    assert "object" in errors[0].lower()


def test_validate_params_enum():
    """Enum constraint: value must be one of the listed options."""

    class _EnumTool(ToolBase):
        name = "etool"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
        }

    t = _EnumTool()
    assert t.validate_params({"mode": "a"}) == []
    errors = t.validate_params({"mode": "c"})
    assert any("one of" in e for e in errors)


def test_validate_params_minimum_maximum():
    """Numeric min/max bounds checked when present."""

    class _BoundedTool(ToolBase):
        name = "bnd"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 10}},
        }

    t = _BoundedTool()
    assert t.validate_params({"n": 5}) == []
    assert any(">=" in e for e in t.validate_params({"n": 0}))
    assert any("<=" in e for e in t.validate_params({"n": 11}))


def test_validate_params_string_length():
    """``minLength`` / ``maxLength`` checked for string fields."""

    class _StrTool(ToolBase):
        name = "stool"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"s": {"type": "string", "minLength": 2, "maxLength": 5}},
        }

    t = _StrTool()
    assert t.validate_params({"s": "abc"}) == []
    assert any("at least" in e for e in t.validate_params({"s": "a"}))
    assert any("at most" in e for e in t.validate_params({"s": "abcdef"}))


def test_validate_params_nested_object():
    """Nested ``object`` schemas recurse — required fields inside the
    nested object are checked too."""

    class _NestedTool(ToolBase):
        name = "ntool"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                }
            },
            "required": ["user"],
        }

    t = _NestedTool()
    assert t.validate_params({"user": {"id": 1}}) == []
    errors = t.validate_params({"user": {}})
    assert any("missing" in e.lower() for e in errors)
