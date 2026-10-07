from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gg.auth.config import IDENTITY_FIELDS, KeyDefaults, KeysConfig, deep_merge, load_keys
from gg.auth.keys import generate_key
from gg.config.loader import ConfigError
from gg.core.keypolicy import KeyPolicy
from tests.unit.api.fakes import KEYS_FILE


def _entry(key_id: str = "app", **extra: Any) -> dict[str, Any]:
    generated = generate_key("test")
    return {
        "id": key_id,
        "name": key_id,
        "hash": generated.hash,
        "prefix": generated.prefix,
        "created_at": "2026-10-05",
        **extra,
    }


def test_deep_merge_rules() -> None:
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2], "s": 1}
    merged = deep_merge(base, {"a": {"y": 3}, "l": [9], "s": 2})
    assert merged == {"a": {"x": 1, "y": 3}, "l": [9], "s": 2}
    assert base == {"a": {"x": 1, "y": 2}, "l": [1, 2], "s": 1}


def test_defaults_deep_merged_into_keys() -> None:
    config = KeysConfig.model_validate(
        {
            "defaults": {
                "allowed_models": ["gg/*", "mock/*"],
                "rate_limits": {"rpm": 10, "tpm": 1000},
                "budget": {"daily_usd": "1.00"},
            },
            "keys": [
                _entry("one", allowed_models=["gg/weak"], rate_limits={"rpm": 99}),
                _entry("two"),
            ],
        }
    )
    one, two = config.keys
    assert one.allowed_models == ("gg/weak",)
    assert one.rate_limits.rpm == 99
    assert one.rate_limits.tpm == 1000
    assert str(one.budget.daily_usd) == "1.00"
    assert two.allowed_models == ("gg/*", "mock/*")
    assert two.rate_limits.rpm == 10


def test_policies_by_hash_strip_hash() -> None:
    entry = _entry()
    policies = KeysConfig.model_validate({"keys": [entry]}).policies_by_hash()
    policy = policies[entry["hash"]]
    assert type(policy) is KeyPolicy
    assert "hash" not in policy.model_dump()


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"keys": [_entry("dup"), _entry("dup")]}, "duplicate key id"),
        (
            {"keys": [_entry("aaa", hash="sha256:" + "a" * 64), _entry("bbb", hash="sha256:" + "a" * 64)]},
            "share a hash",
        ),
        ({"keys": [_entry(hash="md5:abc")]}, "String should match pattern"),
        ({"keys": [_entry(prefix="sk-1234")]}, "String should match pattern"),
        ({"keys": [_entry(allowed_modles=["gg/*"])]}, "Extra inputs are not permitted"),
        ({"defaults": {"rate_limit": {}}, "keys": []}, "Extra inputs are not permitted"),
        ({"version": 2, "keys": []}, "Input should be 1"),
        ({"keys": [_entry(expires_at="2027-01-01T00:00:00")]}, "timezone"),
    ],
)
def test_invalid_configs(data: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        KeysConfig.model_validate(data)


def test_defaults_cover_every_policy_field() -> None:
    assert set(KeyDefaults.model_fields) == set(KeyPolicy.model_fields) - IDENTITY_FIELDS


def test_load_fixture_file() -> None:
    config = load_keys(KEYS_FILE)
    ids = [k.id for k in config.keys]
    assert ids == ["demo", "restricted", "disabled", "expired"]
    restricted = config.keys[1]
    assert restricted.limits.max_completion_tokens == 100
    assert restricted.limits.max_n == 1
    assert restricted.allowed_providers == ("mock",)


def test_repo_keys_file_is_valid() -> None:
    config = load_keys(Path(__file__).resolve().parents[3] / "config" / "keys.yaml")
    assert [k.id for k in config.keys] == ["demo"]


def test_load_missing_file_has_hint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="gg keys create"):
        load_keys(tmp_path / "keys.yaml")


def test_load_invalid_file_reports_location(tmp_path: Path) -> None:
    path = tmp_path / "keys.yaml"
    path.write_text("version: 1\nkeys:\n  - id: x\n")
    with pytest.raises(ConfigError) as info:
        load_keys(path)
    assert any(p.path.startswith("keys.0") for p in info.value.problems)
