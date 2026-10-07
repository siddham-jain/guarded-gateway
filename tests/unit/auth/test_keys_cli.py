import io

import pytest
import yaml

from gg.auth.config import KeysConfig
from gg.auth.keys import hash_key, is_well_formed
from gg.cli.keys_cmd import run
from gg.cli.main import build_parser, main


def _run(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    args = build_parser().parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    code = run(args, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_create_prints_token_once_and_entry() -> None:
    code, out, err = _run(["keys", "create", "--id", "my-app", "--name", "My app", "--env", "test"])
    assert code == 0
    token = out.strip()
    assert out.count("\n") == 1
    assert is_well_formed(token)
    assert token.startswith("gg-test-")
    assert token not in err
    entries = yaml.safe_load(err.split("\n", 1)[1])
    assert entries[0]["hash"] == hash_key(token)
    assert entries[0]["prefix"] == token[:12]
    config = KeysConfig.model_validate({"keys": entries})
    assert config.keys[0].id == "my-app"


def test_create_rejects_bad_id() -> None:
    code, out, err = _run(["keys", "create", "--id", "Bad Id", "--name", "x"])
    assert code == 1
    assert out == ""
    assert "id" in err


def test_create_requires_flags() -> None:
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(["keys", "create", "--name", "x"])
    assert info.value.code == 2


def test_hash_from_stdin() -> None:
    _, token, _ = _run(["keys", "create", "--id", "abc", "--name", "x"])
    code, out, _ = _run(["keys", "hash"], stdin=token)
    assert code == 0
    assert out.strip() == hash_key(token.strip())


def test_hash_rejects_garbage() -> None:
    code, out, err = _run(["keys", "hash"], stdin="sk-nope\n")
    assert code == 1
    assert out == ""
    assert "well-formed" in err


def test_main_dispatches_keys(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["keys", "create", "--id", "abc", "--name", "x"]) == 0
    assert is_well_formed(capsys.readouterr().out.strip())
