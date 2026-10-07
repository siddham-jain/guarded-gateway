from decimal import Decimal
from pathlib import Path

from gg.api.middleware import ADMIN_TAG
from gg.auth.config import load_keys

EXAMPLE = Path(__file__).resolve().parents[3] / "deploy" / "keys.hosted.example.yaml"


def test_hosted_keys_example_is_valid_and_bounded() -> None:
    keys = load_keys(EXAMPLE).keys
    assert all(k.budget.monthly_usd is not None for k in keys)
    # every key maxed out stays under the $3 demo exposure ceiling
    assert sum(k.budget.monthly_usd or Decimal(0) for k in keys) <= Decimal("3.00")
    assert any(ADMIN_TAG in k.tags for k in keys)
    graders = [k for k in keys if k.id.startswith("grader-")]
    assert graders
    assert all(k.limits.max_completion_tokens == 512 and k.expires_at is not None for k in graders)
    assert all(not k.allows_model("gg/demo-chaos") for k in graders)
