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


def test_cast_params_invalid_int_string_passes_through():
    """``int(value)`` raising ValueError → return original value
    unchanged. Validation will reject it later as ``should be integer``."""
    cast_result = _IntTool().cast_params({"n": "not-a-number"})
    assert cast_result == {"n": "not-a-number"}


def test_cast_params_invalid_number_string_passes_through():
    """``float(value)`` raising ValueError → return original value."""

    class _NumTool(ToolBase):
        name = "ntool"
        description = "x"
        parameters = {"type": "object", "properties": {"x": {"type": "number"}}}

    cast_result = _NumTool().cast_params({"x": "abc"})
    assert cast_result == {"x": "abc"}


def test_cast_params_nested_object_recursion():
    """Nested object schemas recurse through ``_cast_object`` so
    inner casts (e.g. string → int) apply on inner fields."""

    class _NestedTool(ToolBase):
        name = "ntool"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            },
        }

    cast_result = _NestedTool().cast_params({"user": {"id": "42"}})
    assert cast_result == {"user": {"id": 42}}


def test_validate_params_array_items():
    """Array items with a per-item schema → each item validated
    against that schema. Wrong-type items show up as errors."""

    class _ArrTool(ToolBase):
        name = "atool"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
        }

    t = _ArrTool()
    assert t.validate_params({"ids": [1, 2, 3]}) == []
    errors = t.validate_params({"ids": [1, "bad"]})
    assert any("integer" in e.lower() for e in errors)


def test_validate_params_number_wrong_type():
    """Number field given a non-numeric value → ``should be number``."""

    class _NumTool(ToolBase):
        name = "ntool"
        description = "x"
        parameters = {"type": "object", "properties": {"x": {"type": "number"}}}

    errors = _NumTool().validate_params({"x": "abc"})
    assert any("number" in e.lower() for e in errors)


def test_cast_params_non_object_schema_short_circuits():
    """If the tool's top-level schema isn't ``type: object``,
    ``cast_params`` returns the input unchanged. Edge case for
    tools with primitive-typed top-level schemas (rare)."""

    class _StringTool(ToolBase):
        name = "st"
        description = "x"
        parameters = {"type": "string"}  # not object — short-circuit

    out = _StringTool().cast_params({"k": "v"})
    assert out == {"k": "v"}


def test_cast_object_with_non_dict_input():
    """``_cast_object`` returns the input unchanged when the input
    isn't a dict — shouldn't crash on an unexpected shape from the
    LLM."""

    class _T(ToolBase):
        name = "t"
        description = "x"
        parameters = {"type": "object", "properties": {}}

    # Internal call with a non-dict value.
    out = _T()._cast_object("not-a-dict", {"type": "object"})
    assert out == "not-a-dict"


def test_cast_object_with_non_dict_properties():
    """If ``schema.properties`` isn't a dict (corrupt schema),
    ``_cast_object`` returns the input dict unchanged."""

    class _T(ToolBase):
        name = "t"
        description = "x"
        parameters = {"type": "object", "properties": "bogus"}

    out = _T().cast_params({"k": "v"})
    assert out == {"k": "v"}


def test_cast_value_with_no_type():
    """``_cast_value`` returns the input unchanged when the schema
    doesn't have a ``type`` key (or it's not a string)."""

    class _T(ToolBase):
        name = "t"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"k": {}},  # no type
        }

    out = _T().cast_params({"k": "v"})
    assert out == {"k": "v"}


def test_cast_value_unknown_target_type_passes_through():
    """Unknown ``type`` value (not in ``_TYPE_MAP``) → passthrough.
    Reaches the final ``return val`` in ``_cast_value``."""

    class _T(ToolBase):
        name = "t"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"k": {"type": "exotic"}},
        }

    out = _T().cast_params({"k": "x"})
    assert out == {"k": "x"}


def test_cast_value_correct_primitive_types_passthrough():
    """``_cast_value`` short-circuits for already-correct primitive
    types — boolean, integer, string, number — without coercing.
    Covers each ``return val`` early-exit branch."""

    class _BoolT(ToolBase):
        name = "b"
        description = "x"
        parameters = {"type": "object", "properties": {"k": {"type": "boolean"}}}

    class _IntT(ToolBase):
        name = "i"
        description = "x"
        parameters = {"type": "object", "properties": {"k": {"type": "integer"}}}

    class _NumT(ToolBase):
        name = "n"
        description = "x"
        parameters = {"type": "object", "properties": {"k": {"type": "number"}}}

    assert _BoolT().cast_params({"k": True}) == {"k": True}
    assert _IntT().cast_params({"k": 5}) == {"k": 5}
    assert _NumT().cast_params({"k": 3.5}) == {"k": 3.5}


def test_cast_array_without_items_schema_passthrough():
    """``array`` schema without ``items`` → return the list as-is
    (no per-item coercion)."""

    class _T(ToolBase):
        name = "t"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"k": {"type": "array"}},
        }

    out = _T().cast_params({"k": [1, "two", 3.5]})
    assert out == {"k": [1, "two", 3.5]}


def test_validate_params_string_field_wrong_type():
    """A string-typed field given a non-string value → ``should be
    string`` error. Hits the generic ``not in (integer, number)``
    type-check branch in ``_validate``."""

    class _StrTool(ToolBase):
        name = "st"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"k": {"type": "string"}},
        }

    errors = _StrTool().validate_params({"k": 42})
    assert any("string" in e.lower() for e in errors)


def test_validate_params_array_field_wrong_type():
    """``array`` field given a non-list → wrong-type error."""

    class _ArrTool(ToolBase):
        name = "at"
        description = "x"
        parameters = {
            "type": "object",
            "properties": {"k": {"type": "array"}},
        }

    errors = _ArrTool().validate_params({"k": "not-a-list"})
    assert any("array" in e.lower() for e in errors)


def test_validate_params_non_object_schema_raises():
    """``validate_params`` raises ``ValueError`` when the top-level
    schema isn't ``type: object`` — that's a misconfigured tool, not
    a runtime input issue, so it raises rather than returning errors."""

    class _BrokenT(ToolBase):
        name = "br"
        description = "x"
        parameters = {"type": "string"}

    raised = False
    try:
        _BrokenT().validate_params({})
    except ValueError as e:
        raised = True
        assert "object" in str(e).lower()
    assert raised


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
