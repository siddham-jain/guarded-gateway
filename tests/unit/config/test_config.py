from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from gg.config.hashing import combined_hash, section_hash
from gg.config.loader import ConfigBundle, ConfigError, ConfigFile, ConfigProblem, load_bundle
from gg.config.settings import Settings
from gg.config.yaml_loader import load_yaml_text


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: str
    price: float = 1.0


def test_only_true_false_are_booleans() -> None:
    data = load_yaml_text("a: off\nb: yes\nc: true\nd: false\ne: on")
    assert data == {"a": "off", "b": "yes", "c": True, "d": False, "e": "on"}


def test_duplicate_keys_raise() -> None:
    with pytest.raises(yaml.YAMLError, match="duplicate key"):
        load_yaml_text("a: 1\na: 2")


def test_bundle_collects_all_problems(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text("mode: x\ntypo: 1\n")
    files = [
        ConfigFile("one", "one.yaml", Sample),
        ConfigFile("two", "two.yaml", Sample),
    ]
    with pytest.raises(ConfigError) as exc:
        load_bundle(files, [], base_dir=tmp_path)
    messages = [str(p) for p in exc.value.problems]
    assert any("typo" in m for m in messages)
    assert any("two.yaml" in m and "not found" in m for m in messages)


def test_optional_files_and_cross_validators(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text("mode: x\n")

    def validator(bundle: ConfigBundle) -> list[ConfigProblem]:
        return [ConfigProblem("one.yaml", "mode", "bad mode")] if bundle.section(Sample).mode == "x" else []

    files = [ConfigFile("one", "one.yaml", Sample), ConfigFile("opt", "opt.yaml", Sample, required=False)]
    with pytest.raises(ConfigError, match="bad mode"):
        load_bundle(files, [validator], base_dir=tmp_path)
    bundle = load_bundle(files[:1] + files[1:], [], base_dir=tmp_path)
    assert bundle.section(Sample).mode == "x"


def test_hash_ignores_formatting_but_tracks_values(tmp_path: Path) -> None:
    a = Sample.model_validate(load_yaml_text("mode: x\nprice: 1.0"))
    b = Sample.model_validate(load_yaml_text("# comment\nprice:   1.0\nmode: x\n"))
    c = Sample.model_validate(load_yaml_text("mode: x\nprice: 2.0"))
    assert section_hash(a) == section_hash(b) != section_hash(c)
    assert len(combined_hash({"a": "1", "b": "2"})) == 12


def test_settings_read_only_gg_prefixed_nested_provider_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
    monkeypatch.setenv("GG_PROVIDERS__TOGETHER__API_KEY", "tg-key")
    monkeypatch.setenv("GG_PROVIDERS__OPENAI__BASE_URL", "http://mock:9000/v1")
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    together = settings.provider("together").api_key
    assert together is not None
    assert together.get_secret_value() == "tg-key"
    assert settings.provider("openai").base_url == "http://mock:9000/v1"
    assert settings.provider("openai").api_key is None
    assert "tg-key" not in repr(settings)
