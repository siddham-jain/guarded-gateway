"""jev ("system one") as the semantic cache's verifier: one noul question about a pair of prompts"""

import asyncio
from typing import Any

import httpx2

from gg.cache.config import VerifierConfig
from gg.core.jsonutil import loads

QUESTION = "same_answer"

QUESTIONS: dict[str, Any] = {
    QUESTION: {
        "type": "noul",
        "instructions": (
            "A correct, complete answer written for `cached_prompt` would also be a correct, complete answer "
            "to `new_prompt`, with nothing missing, extra or wrong. Judge what is asked, not how similar the "
            "wording is."
        ),
        "criteria": {
            "true": (
                "Both prompts ask for the same thing: same subject, same entities and numbers in the same "
                "roles, same polarity, same scope, length, audience, language and output format. Only "
                "wording, word order, spelling, casing or politeness differ."
            ),
            "false": (
                "They differ in any entity, number, unit, direction, negation or antonym, requested format, "
                "length, audience or language, or one asks a narrower, broader or different question about "
                "the same topic."
            ),
        },
    }
}


class VerifierError(Exception):
    """jev could not decide; the lookup is a miss"""


class JevPairVerifier:
    def __init__(self, http: httpx2.AsyncClient, api_key: str, cfg: VerifierConfig) -> None:
        self._http = http
        self._cfg = cfg
        self._path = httpx2.URL(cfg.url).path
        self._headers = {"authorization": f"Bearer {api_key}"}

    async def score(self, cached_prompt: str, new_prompt: str, /) -> float:
        body = {
            "model": self._cfg.model,
            "state": {"cached_prompt": cached_prompt, "new_prompt": new_prompt},
            "questions": QUESTIONS,
        }
        try:
            async with asyncio.timeout(self._cfg.deadline_s):
                response = await self._http.post(self._path, json=body, headers=self._headers)
        except (TimeoutError, httpx2.HTTPError) as exc:
            raise VerifierError(type(exc).__name__) from exc
        if response.status_code != 200:
            raise VerifierError(f"status {response.status_code}")
        try:
            score = loads(response.content)["answers"][QUESTION]["noul"]
        except (ValueError, KeyError, TypeError) as exc:
            raise VerifierError("unexpected response shape") from exc
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise VerifierError("score is not a number")
        return float(score)

    async def same_answer(self, cached_prompt: str, new_prompt: str, /) -> bool:
        return await self.score(cached_prompt, new_prompt) >= self._cfg.min_score
