"""a small json schema subset (type, properties, required, items, enum, const, bounds, combinators).

no json schema library is installed; keywords outside the subset are ignored, so this can only under-report.
"""

from collections.abc import Mapping
from typing import Any, cast

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}
_MAX_ERRORS = 5


def _type_ok(value: Any, name: str) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    if name == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _TYPES.get(name, (object,)))


def _check(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if len(errors) >= _MAX_ERRORS:
        return
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(cast("list[str]", expected))
        if not any(_type_ok(value, n) for n in names):
            errors.append(f"{path}: expected {'|'.join(names)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of the allowed values")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: does not equal the constant")
    for key in ("allOf", "anyOf", "oneOf"):
        subs = cast("list[Mapping[str, Any]]", schema.get(key) or [])
        if not subs:
            continue
        passing = sum(1 for s in subs if not validate(value, s))
        if (
            (key == "allOf" and passing < len(subs))
            or (key == "anyOf" and passing == 0)
            or (key == "oneOf" and passing != 1)
        ):
            errors.append(f"{path}: fails {key}")
    if isinstance(value, dict):
        obj = cast("dict[str, Any]", value)
        props = cast("Mapping[str, Mapping[str, Any]]", schema.get("properties") or {})
        required = cast("list[str]", schema.get("required") or [])
        errors.extend(f"{path}: missing required property '{name}'" for name in required if name not in obj)
        extra = schema.get("additionalProperties", True)
        for name, item in obj.items():
            if name in props:
                _check(item, props[name], f"{path}.{name}", errors)
            elif extra is False:
                errors.append(f"{path}: unexpected property '{name}'")
            elif isinstance(extra, Mapping):
                _check(item, cast("Mapping[str, Any]", extra), f"{path}.{name}", errors)
    elif isinstance(value, list):
        items = cast("list[Any]", value)
        if "minItems" in schema and len(items) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(items) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for i, item in enumerate(items):
                _check(item, cast("Mapping[str, Any]", item_schema), f"{path}[{i}]", errors)
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    elif isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")


def validate(value: Any, schema: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    _check(value, schema, "$", errors)
    return errors
