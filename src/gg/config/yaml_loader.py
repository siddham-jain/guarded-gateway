import re
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

MAX_CONFIG_BYTES = 1024 * 1024


class StrictSafeLoader(yaml.SafeLoader):
    """safe loader where only true/false are booleans and duplicate keys raise"""

    def construct_mapping(self, node: Node, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None, f"expected a mapping, got {type(node).__name__}", node.start_mark
            )
        seen: dict[Any, Node] = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConstructorError(
                    "while constructing a mapping",
                    seen[key].start_mark,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen[key] = key_node
        return super().construct_mapping(node, deep=deep)


StrictSafeLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
StrictSafeLoader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf"))


def load_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=StrictSafeLoader)  # noqa: S506


def load_yaml_file(path: Path) -> Any:
    data = path.read_bytes()
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError(f"{path} is larger than {MAX_CONFIG_BYTES} bytes")
    return load_yaml_text(data.decode("utf-8"))
