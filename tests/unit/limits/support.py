from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import fakeredis

from gg.core.aio import Deadline
from gg.core.clock import FakeClock
from gg.core.context import RequestContext, StageTimings
from gg.core.deployment import Deployment, PriceSchedule, Pricing
from gg.core.schema import ChatRequest
from gg.limits.base import BackendOp, BudgetEvent, RejectionLimit
from gg.providers.base import DeploymentRef, PublicModel, Resolution
from tests.conftest import make_key, make_request

# 2026-10-05 23:00 utc: an hour before the day rolls over, mid-month
WALL = datetime(2026, 10, 5, 23, 0, tzinfo=UTC).timestamp()


def fake_clock(wall: float = WALL) -> FakeClock:
    return FakeClock(wall=wall)


def priced(dep_id: str, input_usd: str, output_usd: str, **pricing: Any) -> Deployment:
    provider = dep_id.split("/", 1)[0]
    period = Pricing(
        effective_from=date(2026, 1, 1), input=Decimal(input_usd), output=Decimal(output_usd), **pricing
    )
    return Deployment(
        id=dep_id,
        provider=provider,
        upstream_model=dep_id.split("/", 1)[1],
        pricing=PriceSchedule(periods=(period,)),
    )


LUNA = priced("openai/gpt-6-luna", "0.10", "0.50")
SOL = priced("openai/gpt-6.1-sol", "2.00", "10.00")
HAIKU = priced("anthropic/claude-haiku-4-5", "1.00", "5.00", cache_write=Decimal("1.25"))


class FakeCatalog:
    """deployments_for over a fixed alias table; enough of ModelCatalog for the limits stage"""

    def __init__(self, aliases: dict[str, Sequence[Deployment]]) -> None:
        self._aliases = aliases

    @property
    def hash(self) -> str:
        return "test"

    def resolve(self, model: str, /) -> Resolution | None:
        deps = self._aliases.get(model)
        return DeploymentRef(deps[0]) if deps else None

    def get(self, deployment_id: str, /) -> Deployment:
        raise KeyError(deployment_id)

    def chain(self, group: str, request: ChatRequest, /) -> list[Deployment]:
        return list(self._aliases.get(group, ()))

    def groups_of(self, alias: str, /) -> tuple[str, ...]:
        return ()

    def deployments_for(self, model: str, /) -> list[Deployment]:
        return list(self._aliases.get(model, ()))

    def list_public(self, allowed: Callable[[str], bool], /) -> Sequence[PublicModel]:
        return ()


class RecordingHooks:
    def __init__(self) -> None:
        self.rejections: list[RejectionLimit] = []
        self.events: list[BudgetEvent] = []
        self.errors: list[BackendOp] = []
        self.degraded_flags: list[bool] = []
        self.expired = 0

    def rejected(self, limit: RejectionLimit, /) -> None:
        self.rejections.append(limit)

    def budget_event(self, event: BudgetEvent, /) -> None:
        self.events.append(event)

    def backend_error(self, op: BackendOp, /) -> None:
        self.errors.append(op)

    def degraded(self, active: bool, /) -> None:
        self.degraded_flags.append(active)

    def holds_expired(self, count: int, /) -> None:
        self.expired += count


def make_ctx(
    clock: FakeClock,
    *,
    key: dict[str, Any] | None = None,
    deadline_s: float = 30,
    **request: Any,
) -> RequestContext:
    req: ChatRequest = make_request(**request)
    return RequestContext(
        request_id="req_test",
        received_at=clock.monotonic(),
        received_unix=int(clock.time()),
        key=make_key(**(key or {})),
        original=req,
        request=req,
        deadline=Deadline.after(deadline_s, clock),
        timings=StageTimings(clock),
    )


def fake_redis(server: fakeredis.FakeServer | None = None) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())


def down_redis() -> fakeredis.FakeAsyncRedis:
    server = fakeredis.FakeServer()
    server.connected = False
    return fakeredis.FakeAsyncRedis(server=server)
