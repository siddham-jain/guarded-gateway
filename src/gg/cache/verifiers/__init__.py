from collections.abc import Callable

import httpx2

from gg.cache.base import SemanticVerifier
from gg.cache.config import VerifierConfig
from gg.cache.verifiers.jev import JevPairVerifier


def build_verifier(
    cfg: VerifierConfig, http_client: Callable[[str], httpx2.AsyncClient], jev_api_key: str | None
) -> SemanticVerifier | None:
    """None when no verifier is configured or its key is missing; the caller decides what that means"""
    if cfg.type == "none" or not jev_api_key:
        return None
    url = httpx2.URL(cfg.url)
    return JevPairVerifier(http_client(f"{url.scheme}://{url.netloc.decode()}"), jev_api_key, cfg)
