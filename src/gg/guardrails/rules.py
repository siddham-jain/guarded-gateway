"""versioned rule packs (config/policies/rules/*.yaml): schemas, compilation and digests"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from gg.config.yaml_loader import load_yaml_file
from gg.core.jsonutil import sha256_hex
from gg.core.schema import StrictModel

type View = Literal["inspect", "decoded"]
type _SemVer = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]

# a match that is still growing at the end of a stream window continues over token characters
RUN_CONTINUATION = r"[A-Za-z0-9_\-+/=.]*"


def _compiles(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    return pattern


class RuleExamples(StrictModel):
    match: tuple[str, ...] = Field(min_length=1)
    no_match: tuple[str, ...] = Field(min_length=1)


class InjectionRule(StrictModel):
    id: Annotated[str, Field(pattern=r"^[A-Z]{3}-[A-Z]{3}-\d{3}$")]
    tags: tuple[str, ...] = Field(min_length=1)
    pattern: str
    views: tuple[View, ...] = ("inspect", "decoded")
    examples: RuleExamples
    notes: str = ""

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: str) -> str:
        return _compiles(value)


class InjectionPackDoc(StrictModel):
    id: str
    version: _SemVer
    engine: Literal["re"] = "re"
    rules: tuple[InjectionRule, ...] = Field(min_length=1)


class SecretRule(StrictModel):
    id: Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]
    pattern: str
    group: Annotated[int, Field(ge=0)] = 0
    # the rule only fires when one of these words appears shortly before the match
    keywords: tuple[str, ...] = ()
    continuation: str = RUN_CONTINUATION
    examples: RuleExamples

    @field_validator("pattern", "continuation")
    @classmethod
    def _check_pattern(cls, value: str) -> str:
        return _compiles(value)


class SecretsPackDoc(StrictModel):
    id: str
    version: _SemVer
    engine: Literal["re"] = "re"
    keyword_window: Annotated[int, Field(ge=1, le=200)] = 40
    # documented example values (AKIAIOSFODNN7EXAMPLE, ...) are never secrets
    allow_substrings: tuple[str, ...] = ()
    # words that make a nearby high-entropy token a secret
    entropy_keywords: tuple[str, ...] = ()
    rules: tuple[SecretRule, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class CompiledInjectionRule:
    id: str
    tags: frozenset[str]
    regex: re.Pattern[str]
    views: frozenset[View]


@dataclass(frozen=True, slots=True)
class InjectionPack:
    doc: InjectionPackDoc
    digest: str
    rules: tuple[CompiledInjectionRule, ...]


@dataclass(frozen=True, slots=True)
class CompiledSecretRule:
    id: str
    regex: re.Pattern[str]
    group: int
    keywords: re.Pattern[str] | None
    continuation: str


@dataclass(frozen=True, slots=True)
class SecretsPack:
    doc: SecretsPackDoc
    digest: str
    rules: tuple[CompiledSecretRule, ...]
    entropy_keywords: re.Pattern[str] | None


def _keyword_regex(words: tuple[str, ...]) -> re.Pattern[str] | None:
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)


class RulePackError(Exception):
    pass


class RulePackStore:
    """loads each pack once; packs are shared by every effective policy that references them"""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._injection: dict[str, InjectionPack] = {}
        self._secrets: dict[str, SecretsPack] = {}

    def _read(self, ref: str) -> tuple[object, str]:
        path = (self._base / ref).resolve()
        if not path.is_relative_to(self._base.resolve()) or not path.is_file():
            raise RulePackError(f"rule pack '{ref}' not found under {self._base}")
        return load_yaml_file(path), sha256_hex(path.read_bytes())

    def injection(self, ref: str) -> InjectionPack:
        pack = self._injection.get(ref)
        if pack is None:
            raw, digest = self._read(ref)
            doc = InjectionPackDoc.model_validate(raw)
            rules = tuple(
                CompiledInjectionRule(r.id, frozenset(r.tags), re.compile(r.pattern), frozenset(r.views))
                for r in doc.rules
            )
            pack = self._injection[ref] = InjectionPack(doc, digest, rules)
        return pack

    def secrets(self, ref: str) -> SecretsPack:
        pack = self._secrets.get(ref)
        if pack is None:
            raw, digest = self._read(ref)
            doc = SecretsPackDoc.model_validate(raw)
            rules = tuple(
                CompiledSecretRule(
                    r.id, re.compile(r.pattern), r.group, _keyword_regex(r.keywords), r.continuation
                )
                for r in doc.rules
            )
            pack = self._secrets[ref] = SecretsPack(doc, digest, rules, _keyword_regex(doc.entropy_keywords))
        return pack

    def digest(self, ref: str) -> str:
        if ref in self._injection:
            return self._injection[ref].digest
        if ref in self._secrets:
            return self._secrets[ref].digest
        return self._read(ref)[1]
