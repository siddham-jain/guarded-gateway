from datetime import date
from decimal import Decimal

from gg.core.deployment import Deployment, PriceSchedule, Pricing
from gg.core.schema import AssistantMessage, ChatResponse, Choice
from gg.core.usage import UsageRecord
from gg.observability.metrics import Metrics

RESPONSE = ChatResponse(
    id="chatcmpl-1",
    created=1,
    model="gpt-6-luna",
    choices=(Choice(index=0, message=AssistantMessage(content="secret reply"), finish_reason="stop"),),
)

LUNA = Deployment(
    id="openai/gpt-6-luna",
    provider="openai",
    upstream_model="gpt-6-luna",
    tier="weak",
    pricing=PriceSchedule(
        periods=(
            Pricing(
                effective_from=date(2026, 1, 1),
                input=Decimal("0.10"),
                output=Decimal("0.50"),
                cached_input=Decimal("0.01"),
            ),
        )
    ),
)


def luna_usage(*, input_tokens: int = 500, output_tokens: int = 200) -> UsageRecord:
    return UsageRecord(
        provider="openai",
        deployment_id=LUNA.id,
        upstream_model="gpt-6-luna",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def new_metrics() -> Metrics:
    return Metrics(
        aliases=["gg/auto", "gg/resilient"],
        deployments=[LUNA.id, "anthropic/claude-haiku-4-5"],
        process_collectors=False,
    )
