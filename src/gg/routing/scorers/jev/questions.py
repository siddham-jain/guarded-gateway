from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from gg.core.jsonutil import canonical_json, loads, sha256_hex


class QuestionSetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QuestionSet:
    version: str
    questions: Mapping[str, Any]
    tier_options: tuple[str, ...]
    content_hash: str


def qset_path(config_dir: Path, name: str) -> Path:
    return config_dir / "routing" / "qsets" / f"{name}.json"


def load_question_set(config_dir: Path, name: str, strong_tiers: Sequence[str]) -> QuestionSet:
    path = qset_path(config_dir, name)
    try:
        raw = loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise QuestionSetError(f"question set file not found: {path}") from exc
    except ValueError as exc:
        raise QuestionSetError(f"{path}: invalid json: {exc}") from exc
    return parse_question_set(name, raw, strong_tiers, source=str(path))


def parse_question_set(name: str, raw: Any, strong_tiers: Sequence[str], *, source: str) -> QuestionSet:
    if not isinstance(raw, dict):
        raise QuestionSetError(f"{source}: expected a questions object")
    questions = cast("dict[str, Any]", raw)
    tier = questions.get("tier")
    if not isinstance(tier, dict) or tier.get("type") != "choice":
        raise QuestionSetError(f"{source}: needs a 'tier' question of type 'choice'")
    criteria = cast("dict[str, Any]", tier).get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise QuestionSetError(f"{source}: 'tier' needs at least two options in 'criteria'")
    options = tuple(str(k) for k in cast("dict[str, Any]", criteria))
    unknown = sorted(set(strong_tiers) - set(options))
    if unknown:
        raise QuestionSetError(f"{source}: strong_tiers {unknown} are not 'tier' options {list(options)}")
    if set(strong_tiers) == set(options):
        raise QuestionSetError(f"{source}: every 'tier' option is strong; at least one must map to weak")
    return QuestionSet(
        version=name,
        questions=questions,
        tier_options=options,
        content_hash=sha256_hex(canonical_json(questions)),
    )
