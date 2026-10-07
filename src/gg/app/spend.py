from collections.abc import AsyncIterator

from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk, ChatRequest
from gg.core.usage import usage_record_from_chunk
from gg.limits.base import ProviderSpendGuard
from gg.limits.cost import CostCalculator
from gg.providers.base import ProviderAdapter


class SpendGuardedAdapter:
    """decorator: refuses billed attempts once a provider's spend cap is hit and books what each call cost.

    a denial is a provider-scoped fallback error, so the executor moves on to the next provider in the plan.
    """

    def __init__(
        self, inner: ProviderAdapter, guard: ProviderSpendGuard, costs: CostCalculator, clock: Clock
    ) -> None:
        self.name = inner.name
        self._inner = inner
        self._guard = guard
        self._costs = costs
        self._clock = clock

    async def stream(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext, /
    ) -> AsyncIterator[ChatChunk]:
        billed = deployment.pricing.billed
        if billed and (denial := await self._guard.check(deployment.provider)) is not None:
            raise ProviderError(
                "fallback",
                provider=deployment.provider,
                status=0,
                code="spend_cap",
                message=f"spend cap reached ({denial.reason})",
                scope="provider",
                deployment_id=deployment.id,
            )
        last: ChatChunk | None = None
        try:
            async for chunk in self._inner.stream(request, deployment, ctx):
                if chunk.usage is not None:
                    last = chunk
                yield chunk
        finally:
            if billed and last is not None:
                await self._book(last, deployment)

    async def _book(self, chunk: ChatChunk, deployment: Deployment) -> None:
        usage = usage_record_from_chunk(chunk, deployment)
        if usage is None:
            return
        breakdown = self._costs.cost(usage, deployment, self._clock.now())
        if breakdown is not None and breakdown.total > 0:
            await self._guard.record(deployment.provider, breakdown.total)

    async def aclose(self) -> None:
        await self._inner.aclose()
