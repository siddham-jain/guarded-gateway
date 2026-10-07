import pytest

import gg.providers.usage


@pytest.fixture(autouse=True)
def _offline_token_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    # tiktoken downloads its encoding on first use; tests stay offline with the char heuristic
    monkeypatch.setattr(gg.providers.usage, "_encoder", lambda: None)
