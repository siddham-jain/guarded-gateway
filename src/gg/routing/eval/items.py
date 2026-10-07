"""routing eval items (C5 §10.3) and the suite file that freezes the model pairs (C11 §3.8)"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError

from gg.config.loader import load_file
from gg.core.jsonutil import canonical_json, loads, sha256_hex
from gg.core.schema import StrictModel

type Tier = Literal["weak", "strong"]
type Split = Literal["tune", "heldout"]


class NumericGrader(StrictModel):
    type: Literal["numeric"]
    answer: float
    tol: float = 1e-6
    rel_tol: float = 0.0


class ExactGrader(StrictModel):
    """exact: the final answer equals one of answers; word: one answer appears as a word and no reject does"""

    type: Literal["exact"]
    answers: Annotated[tuple[str, ...], Field(min_length=1)]
    match: Literal["exact", "word"] = "exact"
    reject: tuple[str, ...] = ()


class ChoiceGrader(StrictModel):
    type: Literal["choice"]
    answer: Annotated[str, Field(pattern=r"^[A-H]$")]


class RegexGrader(StrictModel):
    type: Literal["regex"]
    all: Annotated[tuple[str, ...], Field(min_length=1)]
    none: tuple[str, ...] = ()
    case_sensitive: bool = False


class JsonGrader(StrictModel):
    """expect is matched as a subset; {"$any": [...]} and {"$contains": "..."} relax string checks"""

    type: Literal["json"]
    expect: Any
    tol: float = 0.01


class ConstraintsGrader(StrictModel):
    """every listed constraint must hold; for format-following prompts"""

    type: Literal["constraints"]
    max_words: int | None = None
    min_words: int | None = None
    exact_words: int | None = None
    max_sentences: int | None = None
    lines: int | None = None
    max_lines: int | None = None
    list_items: int | None = None
    word_initial: str | None = None
    contains_all: tuple[str, ...] = ()
    contains_any: tuple[str, ...] = ()
    contains_none: tuple[str, ...] = ()
    regex: tuple[str, ...] = ()


class JudgeGrader(StrictModel):
    """open-ended: pairwise llm judge, both orders"""

    type: Literal["judge"]
    rubric: str | None = None


type Grader = Annotated[
    NumericGrader | ExactGrader | ChoiceGrader | RegexGrader | JsonGrader | ConstraintsGrader | JudgeGrader,
    Field(discriminator="type"),
]


class EvalMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1)]


class EvalItem(StrictModel):
    id: Annotated[str, Field(pattern=r"^rt\d{3,4}$")]
    category: str
    split: Split
    source: str = "handwritten"
    # the author's prior, used for the dry-run fakes and a side metric; never the routing label
    expected_tier: Tier
    messages: Annotated[tuple[EvalMessage, ...], Field(min_length=1)]
    grader: Grader
    reference: str | None = None
    max_tokens: Annotated[int, Field(ge=16, le=8192)] = 600
    notes: str = ""

    @property
    def needs_judge(self) -> bool:
        return self.grader.type == "judge"

    def wire_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    @property
    def prompt_sha(self) -> str:
        return sha256_hex(canonical_json(self.wire_messages()))

    @property
    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


def load_items(path: Path) -> list[EvalItem]:
    """one json object per line; blank lines skipped; ids must be unique"""
    items: list[EvalItem] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = EvalItem.model_validate(loads(line))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{number}: duplicate item id {item.id}")
        seen.add(item.id)
        items.append(item)
    return items


def limit_items(items: Sequence[EvalItem], limit: int | None) -> list[EvalItem]:
    """first n items taken round-robin across categories, so a small pilot still covers every category"""
    if limit is None or limit >= len(items):
        return list(items)
    queues: dict[str, list[EvalItem]] = {}
    for item in items:
        queues.setdefault(item.category, []).append(item)
    picked: list[EvalItem] = []
    while len(picked) < limit:
        for queue in queues.values():
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    order = {item.id: i for i, item in enumerate(items)}
    return sorted(picked, key=lambda i: order[i.id])


def items_digest(items: Sequence[EvalItem]) -> str:
    payload = [i.model_dump(mode="json") for i in sorted(items, key=lambda i: i.id)]
    return sha256_hex(canonical_json(payload))


class RoleConfig(StrictModel):
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    # added to an item's max_tokens for models whose reasoning tokens count against the cap
    extra_tokens: Annotated[int, Field(ge=0)] = 0
    # judge only: fixed completion cap
    max_tokens: Annotated[int, Field(ge=16)] | None = None


class PairConfig(StrictModel):
    profile: str
    description: str = ""
    weak: RoleConfig
    strong: RoleConfig
    judge: RoleConfig


class SuiteConfig(StrictModel):
    version: Literal[1] = 1
    suite_version: str
    items: Path
    judge_prompt: Path
    pairs: dict[str, PairConfig]


def load_suite(path: Path) -> SuiteConfig:
    suite = load_file(path, SuiteConfig)
    base = path.parent
    return suite.model_copy(update={"items": base / suite.items, "judge_prompt": base / suite.judge_prompt})
