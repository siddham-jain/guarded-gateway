import ssl

import httpx2
import pytest

from gg.config.settings import Settings
from gg.core.clock import SystemClock
from gg.routing.config import JevConfig
from gg.routing.scorers.jev.client import JevClient
from tests.unit.routing.support import make_scorer, routing_request, user

pytestmark = pytest.mark.live


async def test_one_real_jev_call_matches_the_contract() -> None:
    key = Settings().jev_api_key
    if key is None:
        pytest.skip("GG_JEV_API_KEY is not set")
    # a cold tls connection exceeds the 400 ms production budget; this checks shape, not latency
    cfg = JevConfig(attempt_timeout_ms=3000, connect_timeout_ms=2000, max_attempts=1)
    # openssl trust store: the macos keychain path fails inside sandboxed runners
    async with httpx2.AsyncClient(verify=ssl.create_default_context()) as http:
        client = JevClient(http, key.get_secret_value(), cfg, SystemClock())
        scorer = make_scorer(client, cfg, deadline_s=4)
        score = await scorer.score(
            routing_request(
                user("We moved from RabbitMQ to Redis last week."),
                {"role": "assistant", "content": "Noted."},
                user("Our Celery workers randomly hang after ~2 hours with no error. How do I debug it?"),
            )
        )
    print(f"jev live: latency_ms={score.latency_ms} score={score.score:.2f} tier={score.tier}")
    assert not score.fallback, score.fallback_reason
    assert score.response_model == "jev-1.13.0"
    assert score.request_id is not None
    assert score.input_tokens is not None
    assert score.score >= 0.5
    assert set(score.tier_probabilities) == {"small", "mid", "frontier", "frontier_reasoning"}
