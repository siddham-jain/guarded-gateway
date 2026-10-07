from collections.abc import Mapping

from pydantic import BaseModel

from gg.core.jsonutil import canonical_json, sha256_hex


def section_hash(model: BaseModel) -> str:
    # hashing the validated model: formatting and comments don't matter, changed defaults do
    return sha256_hex(canonical_json(model.model_dump(mode="json", by_alias=True)))


def combined_hash(section_hashes: Mapping[str, str], *, length: int = 12) -> str:
    joined = "|".join(f"{name}={section_hashes[name]}" for name in sorted(section_hashes))
    return sha256_hex(joined)[:length]
