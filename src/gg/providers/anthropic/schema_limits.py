from typing import Any, cast

UNSUPPORTED_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
)
MAX_OPTIONAL_PARAMS = 24
MAX_UNION_PARAMS = 16
# keys whose values are name -> schema maps, and keys whose values are data rather than schemas
_NAME_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
_LITERALS = frozenset({"enum", "const", "default", "examples", "required"})


class _Walk:
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.optional = 0
        self.unions = 0

    def add(self, violation: str) -> None:
        if violation not in self.violations:
            self.violations.append(violation)

    def node(self, value: Any) -> None:
        if isinstance(value, list):
            for item in cast("list[Any]", value):
                self.node(item)
            return
        if not isinstance(value, dict):
            return
        schema = cast("dict[str, Any]", value)
        for keyword in UNSUPPORTED_KEYWORDS:
            if keyword in schema:
                self.add(keyword)
        if schema.get("$ref") == "#":
            self.add("recursion")
        if "additionalProperties" in schema and schema["additionalProperties"] is not False:
            self.add("additionalProperties")
        kind = schema.get("type")
        if "anyOf" in schema or (isinstance(kind, list) and len(cast("list[Any]", kind)) > 1):
            self.unions += 1
        properties = schema.get("properties")
        if isinstance(properties, dict):
            names = cast("dict[str, Any]", properties)
            required = schema.get("required")
            required_names = set(cast("list[Any]", required)) if isinstance(required, list) else set()
            self.optional += sum(1 for name in names if name not in required_names)
        for key, child in schema.items():
            if key in _NAME_MAPS and isinstance(child, dict):
                for sub in cast("dict[str, Any]", child).values():
                    self.node(sub)
            elif key not in _LITERALS and isinstance(child, (dict, list)):
                self.node(child)


def schema_violations(schema: Any) -> tuple[str, ...]:
    """anthropic structured-output limits (research §1.6); names the offending keyword"""
    walk = _Walk()
    walk.node(schema)
    if walk.optional > MAX_OPTIONAL_PARAMS:
        walk.add("optional_params")
    if walk.unions > MAX_UNION_PARAMS:
        walk.add("union_params")
    return tuple(walk.violations)
