import string

import pytest

from gg.auth.keys import (
    KEY_RE,
    check_format,
    generate_key,
    hash_key,
    is_well_formed,
    key_prefix,
    parse_bearer,
)
from gg.core.errors import AuthenticationError


@pytest.mark.parametrize("env", ["live", "test"])
def test_generated_key_shape(env: str) -> None:
    key = generate_key(env)  # pyright: ignore[reportArgumentType]
    assert len(key.token) == 57 if env == "live" else 57
    assert KEY_RE.match(key.token)
    assert key.token.startswith(f"gg-{env}-")
    assert is_well_formed(key.token)
    assert key.prefix == key.token[:12]
    assert key.hash == hash_key(key.token)


def test_keys_are_unique() -> None:
    assert len({generate_key().token for _ in range(200)}) == 200


def test_hash_format() -> None:
    h = hash_key("gg-test-x")
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64
    assert all(c in string.hexdigits for c in h[7:])


def test_checksum_rejects_every_single_char_typo() -> None:
    token = generate_key().token
    alphabet = string.digits + string.ascii_letters
    for i in range(8, len(token)):
        for replacement in alphabet[:5]:
            if replacement == token[i]:
                continue
            typo = token[:i] + replacement + token[i + 1 :]
            assert not is_well_formed(typo), (i, replacement)


@pytest.mark.parametrize(
    "token",
    [
        "",
        "gg-live-",
        "gg-prod-" + "a" * 49,
        "gg-live-" + "a" * 48,
        "gg-live-" + "a" * 50,
        "gg-live-" + "!" * 49,
    ],
)
def test_malformed_tokens(token: str) -> None:
    assert not is_well_formed(token)
    with pytest.raises(AuthenticationError) as info:
        check_format(token, allow_test_keys=True)
    assert info.value.code == "invalid_api_key"


def test_test_keys_rejected_when_not_allowed() -> None:
    token = generate_key("test").token
    check_format(token, allow_test_keys=True)
    with pytest.raises(AuthenticationError) as info:
        check_format(token, allow_test_keys=False)
    assert info.value.code == "invalid_api_key"
    check_format(generate_key("live").token, allow_test_keys=False)


def test_prefix_is_display_safe() -> None:
    assert key_prefix("gg-live-7Hq2abcdef") == "gg-live-7Hq2"


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ([], "missing_api_key"),
        (["Bearer gg-first-token", "Bearer gg-second-token"], "invalid_api_key"),
        ([""], "invalid_api_key"),
        (["Bearer"], "invalid_api_key"),
        (["Basic Z2c6Z2c="], "invalid_api_key"),
        (["Bearer gg-a gg-b"], "invalid_api_key"),
        (["Bearer gg-" + "a" * 300], "invalid_api_key"),
        (["Bearer sk-proj-123"], "invalid_api_key"),
    ],
)
def test_parse_bearer_rejects(values: list[str], code: str) -> None:
    with pytest.raises(AuthenticationError) as info:
        parse_bearer(values)
    assert info.value.code == code
    for value in values:
        for token in value.split()[1:]:
            assert token not in info.value.message


def test_parse_bearer_hints_at_gg_keys() -> None:
    with pytest.raises(AuthenticationError, match="start with gg-"):
        parse_bearer(["Bearer sk-abc"])


@pytest.mark.parametrize("header", ["Bearer gg-x", "bearer gg-x", "BEARER   gg-x  ", "  Bearer gg-x"])
def test_parse_bearer_accepts(header: str) -> None:
    assert parse_bearer([header]) == "gg-x"
