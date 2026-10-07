from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from yaml import YAMLError

from gg.config.hashing import section_hash
from gg.config.yaml_loader import load_yaml_file


@dataclass(frozen=True, slots=True)
class ConfigProblem:
    file: str
    path: str
    message: str

    def __str__(self) -> str:
        where = f"{self.file}:{self.path}" if self.path else self.file
        return f"{where}: {self.message}"


class ConfigError(Exception):
    def __init__(self, problems: Sequence[ConfigProblem]) -> None:
        self.problems = tuple(problems)
        super().__init__("\n".join(str(p) for p in self.problems))


@dataclass(frozen=True, slots=True)
class ConfigFile[M: BaseModel]:
    name: str
    path: str
    model: type[M]
    hashed: bool = True
    required: bool = True


@dataclass(frozen=True, slots=True)
class ConfigBundle:
    """sections keyed by schema type, so gg.config never imports feature packages"""

    sections: Mapping[type[BaseModel], BaseModel]
    section_hashes: Mapping[str, str]

    def section[M: BaseModel](self, model: type[M]) -> M:
        value = self.sections.get(model)
        if value is None:
            raise KeyError(f"config section {model.__name__} is not loaded")
        return cast("M", value)

    def optional[M: BaseModel](self, model: type[M]) -> M | None:
        return cast("M | None", self.sections.get(model))


type CrossValidator = Callable[[ConfigBundle], Iterable[ConfigProblem]]


def _validation_problems(file: str, exc: ValidationError) -> list[ConfigProblem]:
    return [
        ConfigProblem(file, ".".join(str(p) for p in err["loc"]), err["msg"])
        for err in exc.errors(include_input=False)
    ]


def load_file[M: BaseModel](path: Path, model: type[M]) -> M:
    raw: Any = load_yaml_file(path)
    return model.model_validate(raw if raw is not None else {})


def load_bundle(
    files: Sequence[ConfigFile[Any]],
    validators: Sequence[CrossValidator],
    *,
    base_dir: Path,
    overrides: Mapping[str, Path] | None = None,
) -> ConfigBundle:
    problems: list[ConfigProblem] = []
    sections: dict[type[BaseModel], BaseModel] = {}
    hashes: dict[str, str] = {}
    for spec in files:
        path = (overrides or {}).get(spec.name) or base_dir / spec.path
        if not path.exists():
            if spec.required:
                problems.append(ConfigProblem(str(path), "", "file not found"))
            continue
        try:
            value = load_file(path, spec.model)
        except ValidationError as exc:
            problems.extend(_validation_problems(str(path), exc))
            continue
        except (YAMLError, ValueError, UnicodeDecodeError) as exc:
            problems.append(ConfigProblem(str(path), "", str(exc)))
            continue
        sections[spec.model] = value
        if spec.hashed:
            hashes[spec.name] = section_hash(value)
    bundle = ConfigBundle(sections, hashes)
    if not problems:
        for validator in validators:
            problems.extend(validator(bundle))
    if problems:
        raise ConfigError(problems)
    return bundle
