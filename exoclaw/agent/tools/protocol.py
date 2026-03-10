"""Tool protocol and optional ToolBase mixin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable


@dataclass
class ToolContext:
    """
    Session context passed to tools that implement execute_with_context().

    Duck-typed — tools that need session routing implement:

        async def execute_with_context(self, ctx: ToolContext, **kwargs: object) -> str:
            ...

    Tools that don't need it keep execute(**kwargs) and the registry handles both.
    """

    session_key: str
    channel: str
    chat_id: str


@runtime_checkable
class Tool(Protocol):
    """
    Structural protocol for agent tools.

    External packages implement this without inheriting from any exoclaw class:

        class MyTool:
            name = "my_tool"
            description = "Does something useful."
            parameters = {"type": "object", "properties": {...}, "required": [...]}

            async def execute(self, **kwargs: object) -> str:
                ...

    Use ToolBase as an optional mixin to get
    cast_params / validate_params / to_schema utilities for free.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, object]: ...

    async def execute(self, **kwargs: object) -> str: ...


class ToolBase:
    """
    Optional mixin for tool authors.

    Provides cast_params, validate_params, and to_schema so you don't have
    to write them yourself. Inherit from ToolBase to get them for free, or
    skip it and just satisfy the Tool protocol directly.
    """

    _TYPE_MAP: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def cast_params(self, params: dict[str, object]) -> dict[str, object]:
        """Apply safe schema-driven casts before validation."""
        schema = self.parameters or {}  # type: ignore[attr-defined]
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: object, schema: dict[str, object]) -> dict[str, object]:
        """Cast an object (dict) according to schema."""
        if not isinstance(obj, dict):
            return cast(dict[str, object], obj)
        props = schema.get("properties", {})
        result: dict[str, object] = {}
        if not isinstance(props, dict):
            return dict(obj)
        for key, value in obj.items():
            if key in props:
                prop_schema = props[key]
                if isinstance(prop_schema, dict):
                    result[key] = self._cast_value(value, prop_schema)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def _cast_value(self, val: object, schema: dict[str, object]) -> object:
        """Cast a single value according to schema."""
        target_type = schema.get("type")

        if target_type == "boolean" and isinstance(val, bool):
            return val
        if target_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if target_type in self._TYPE_MAP and target_type not in (
            "boolean",
            "integer",
            "array",
            "object",
        ):
            expected = self._TYPE_MAP[target_type]
            if isinstance(val, expected):
                return val

        if target_type == "integer" and isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val

        if target_type == "number" and isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val

        if target_type == "string":
            return val if val is None else str(val)

        if target_type == "boolean" and isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("true", "1", "yes"):
                return True
            if val_lower in ("false", "0", "no"):
                return False
            return val

        if target_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                return [self._cast_value(item, item_schema) for item in val]
            return val

        if target_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, object]) -> list[str]:
        """Validate tool parameters against JSON schema. Returns error list (empty if valid)."""
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}  # type: ignore[attr-defined]
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: object, schema: dict[str, object], path: str) -> list[str]:
        t, label = schema.get("type"), path or "parameter"
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, self._TYPE_MAP[t]) or isinstance(val, bool)  # type: ignore[literal-required]
        ):
            return [f"{label} should be number"]
        if (
            t in self._TYPE_MAP
            and t not in ("integer", "number")
            and not isinstance(val, self._TYPE_MAP[t])
        ):  # type: ignore[literal-required]
            return [f"{label} should be {t}"]

        errors = []
        enum = schema.get("enum")
        if enum is not None and isinstance(enum, list) and val not in enum:
            errors.append(f"{label} must be one of {enum}")
        if t in ("integer", "number") and isinstance(val, (int, float)):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and val < minimum:  # type: ignore[operator]
                errors.append(f"{label} must be >= {minimum}")
            if maximum is not None and val > maximum:  # type: ignore[operator]
                errors.append(f"{label} must be <= {maximum}")
        if t == "string" and isinstance(val, str):
            min_length = schema.get("minLength")
            max_length = schema.get("maxLength")
            if min_length is not None and isinstance(min_length, int) and len(val) < min_length:
                errors.append(f"{label} must be at least {min_length} chars")
            if max_length is not None and isinstance(max_length, int) and len(val) > max_length:
                errors.append(f"{label} must be at most {max_length} chars")
        if t == "object" and isinstance(val, dict):
            props = schema.get("properties", {})
            required = schema.get("required", [])
            if isinstance(required, list):
                for k in required:
                    if k not in val:
                        errors.append(f"missing required {path + '.' + k if path else k}")
            if isinstance(props, dict):
                for k, v in val.items():
                    if k in props:
                        prop_schema = props[k]
                        if isinstance(prop_schema, dict):
                            errors.extend(
                                self._validate(v, prop_schema, path + "." + k if path else k)
                            )
        if t == "array" and "items" in schema and isinstance(val, list):
            item_schema = schema["items"]
            if isinstance(item_schema, dict):
                for i, item in enumerate(val):
                    errors.extend(
                        self._validate(item, item_schema, f"{path}[{i}]" if path else f"[{i}]")
                    )
        return errors

    def to_schema(self) -> dict[str, object]:
        """Convert tool to OpenAI function schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,  # type: ignore[attr-defined]
                "description": self.description,  # type: ignore[attr-defined]
                "parameters": self.parameters,  # type: ignore[attr-defined]
            },
        }
