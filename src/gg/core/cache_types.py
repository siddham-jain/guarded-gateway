from dataclasses import dataclass
from typing import Literal

type CacheStatus = Literal["miss", "exact_hit", "semantic_hit", "bypass"]


@dataclass(slots=True)
class CacheState:
    exact_key: str | None = None
    semantic_distance: float | None = None
    bypass_reason: str | None = None
    store: bool = True
