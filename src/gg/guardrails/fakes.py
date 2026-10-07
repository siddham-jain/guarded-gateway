"""seeded generator of obviously fake, validly shaped secrets and pii.

key-shaped literals never live in the repo (push protection and our own scanners would flag them);
rule examples, eval items and tests write `{{fake:github_pat}}` and expand it here.
"""

import random
import re
import string
from collections.abc import Callable

_ALNUM = string.ascii_letters + string.digits
_UPPER_DIGITS = string.ascii_uppercase + "234567"
_TEMPLATE_RE = re.compile(r"\{\{fake:([a-z0-9_]+)(?:\[(\d*):(\d*)\])?\}\}")


def _run(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _luhn_complete(digits: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return digits + str((10 - total % 10) % 10)


def _iban(rng: random.Random) -> str:
    bban = _run(rng, string.digits, 18)
    numeric = "".join(str(int(ch, 36)) for ch in bban + "DE00")
    check = 98 - int(numeric) % 97
    return f"DE{check:02d}{bban}"


def _jwt(rng: random.Random) -> str:
    head = "ey" + "J" + _run(rng, _ALNUM, 30)
    body = "ey" + "J" + _run(rng, _ALNUM, 40)
    return f"{head}.{body}.{_run(rng, _ALNUM + '-_', 43)}"


def _pem(rng: random.Random) -> str:
    kind = "RSA " + "PRIVATE KEY"
    lines = "\n".join(_run(rng, _ALNUM + "+/", 64) for _ in range(4))
    return f"-----BEGIN {kind}-----\n{lines}\n-----END {kind}-----"


_GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "aws_access_key": lambda r: "AK" + "IA" + _run(r, _UPPER_DIGITS, 16),
    "aws_secret_key": lambda r: _run(r, _ALNUM + "/+", 40),
    "github_pat": lambda r: "gh" + "p_" + _run(r, _ALNUM, 36),
    "github_fine_grained": lambda r: "github" + "_pat_" + _run(r, _ALNUM + "_", 82),
    "openai_key": lambda r: "sk" + "-proj-" + _run(r, _ALNUM + "_-", 48),
    "anthropic_key": lambda r: "sk" + "-ant-api03-" + _run(r, _ALNUM + "_-", 60),
    "google_api_key": lambda r: "AI" + "za" + _run(r, _ALNUM + "_-", 35),
    "slack_token": lambda r: "xo" + "xb-" + _run(r, string.digits, 12) + "-" + _run(r, _ALNUM, 24),
    "hf_token": lambda r: "h" + "f_" + _run(r, _ALNUM, 34),
    "stripe_key": lambda r: "sk" + "_live_" + _run(r, _ALNUM, 24),
    "gitlab_pat": lambda r: "gl" + "pat-" + _run(r, _ALNUM, 20),
    "slack_webhook": lambda r: (
        "https://hooks.slack.com/services/"
        + "T"
        + _run(r, string.ascii_uppercase + string.digits, 8)
        + "/B"
        + _run(r, string.ascii_uppercase + string.digits, 8)
        + "/"
        + _run(r, _ALNUM, 24)
    ),
    "password": lambda r: _run(r, _ALNUM, 16),
    "jwt": _jwt,
    "pem_private_key": _pem,
    "high_entropy": lambda r: _run(r, _ALNUM + "+/", 32),
    "email": lambda r: (
        f"{_run(r, string.ascii_lowercase, 6)}.{_run(r, string.ascii_lowercase, 4)}@corp.example"
    ),
    "phone": lambda r: f"+44 20 {_run(r, string.digits, 4)} {_run(r, string.digits, 4)}",
    "card": lambda r: _luhn_complete("4" + _run(r, string.digits, 14)),
    "iban": _iban,
    "ssn": lambda r: f"{r.randint(100, 599)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}",
    "ipv4": lambda r: f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}",
}

KINDS: frozenset[str] = frozenset(_GENERATORS)


class FakeValues:
    """same seed -> same values; one value per (kind, n) so a template repeated in one item stays equal"""

    def __init__(self, seed: int = 7) -> None:
        self._seed = seed
        self._cache: dict[str, str] = {}

    def get(self, kind: str) -> str:
        if kind not in _GENERATORS:
            raise KeyError(f"unknown fake kind '{kind}'; known: {', '.join(sorted(_GENERATORS))}")
        if kind not in self._cache:
            rng = random.Random(f"{self._seed}:{kind}")  # noqa: S311 - fakes, not secrets
            self._cache[kind] = _GENERATORS[kind](rng)
        return self._cache[kind]

    def expand(self, text: str) -> str:
        """`{{fake:kind}}`, or a python slice of it, `{{fake:kind[:6]}}`, to split a value across chunks"""

        def sub(m: re.Match[str]) -> str:
            value = self.get(m.group(1))
            if m.group(2) is None and m.group(3) is None:
                return value
            start = int(m.group(2)) if m.group(2) else None
            end = int(m.group(3)) if m.group(3) else None
            return value[start:end]

        return _TEMPLATE_RE.sub(sub, text)


def fake(kind: str, seed: int = 7) -> str:
    return FakeValues(seed).get(kind)
