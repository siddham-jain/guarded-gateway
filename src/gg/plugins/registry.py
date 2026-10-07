from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError


class RegistryError(Exception):
    pass


type Factory[T, D] = Callable[[Any, D], T]


class Registry[T, D]:
    """named factories with their own config schema; config says `type: name, config: {...}`"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, tuple[Factory[T, D], type[BaseModel]]] = {}

    def register(self, name: str, factory: Factory[T, D], *, config_model: type[BaseModel]) -> None:
        if name in self._factories:
            raise RegistryError(f"{self.kind} '{name}' is already registered")
        self._factories[name] = (factory, config_model)

    def create(self, name: str, raw_config: Mapping[str, Any] | BaseModel, deps: D) -> T:
        entry = self._factories.get(name)
        if entry is None:
            available = ", ".join(sorted(self._factories)) or "none"
            raise RegistryError(f"unknown {self.kind} '{name}'; available: {available}")
        factory, config_model = entry
        if isinstance(raw_config, BaseModel):
            config = raw_config
        else:
            try:
                config = config_model.model_validate(dict(raw_config))
            except ValidationError as exc:
                raise RegistryError(f"invalid config for {self.kind} '{name}': {exc}") from exc
        return factory(config, deps)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return name in self._factories
