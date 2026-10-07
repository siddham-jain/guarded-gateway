"""content-addressed append-only jsonl stores (C11 §3.8); reruns read them back and never pay twice"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from gg.core.jsonutil import canonical_json, dumps, loads, sha256_hex


def record_key(*parts: Any) -> str:
    return sha256_hex(canonical_json(list(parts)))


class JsonlStore:
    """one {"key": ..., ...} object per line; the last line for a key wins; a torn final line is ignored"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, dict[str, Any]] = {}
        if path.is_file():
            self._load()

    def _load(self) -> None:
        lines = self.path.read_bytes().splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = loads(line)
            except ValueError as exc:
                if number == len(lines):
                    # a run killed mid-write leaves a partial last line; that call is simply redone
                    continue
                raise ValueError(f"{self.path}:{number}: corrupt record") from exc
            self._records[record["key"]] = record

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._records.values())

    def put(self, key: str, value: Mapping[str, Any]) -> dict[str, Any]:
        record = {"key": key, **value}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as fh:
            fh.write(dumps(record) + b"\n")
        self._records[key] = record
        return record
