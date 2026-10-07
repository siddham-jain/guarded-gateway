"""paraphrase / near-miss pair schema (C8 §8.1).

labelling rule: should_hit is true iff a correct, complete answer to the anchor is also a correct, complete
answer to the candidate (answer-equivalence, not topical similarity).
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from gg.core.jsonutil import loads
from gg.core.schema import StrictModel

type Category = Literal[
    "paraphrase",
    "surface_noise",
    "politeness",
    "entity_swap",
    "number_change",
    "negation",
    "scope_format",
    "same_topic",
    "short",
]


class Pair(StrictModel):
    id: Annotated[str, Field(pattern=r"^p\d{4}$")]
    anchor: Annotated[str, Field(min_length=1)]
    candidate: Annotated[str, Field(min_length=1)]
    should_hit: bool
    category: Category
    split: Literal["dev", "test"] = "dev"
    source: str = "hand"
    notes: str = ""


def load_pairs(path: Path) -> list[Pair]:
    """one json object per line; blank lines skipped; ids must be unique"""
    pairs: list[Pair] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pair = Pair.model_validate(loads(line))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if pair.id in seen:
            raise ValueError(f"{path}:{number}: duplicate pair id {pair.id}")
        seen.add(pair.id)
        pairs.append(pair)
    return pairs
