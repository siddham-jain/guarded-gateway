from gg.limits.base import BudgetStatus, LimitResult, Micros
from gg.limits.ledger import usd


def go_duration(seconds: float) -> str:
    """openai-style go durations: 0s, 850ms, 1s, 1.5s, 6m0s, 1h0m0s"""
    ms = max(0, round(seconds * 1000))
    if ms == 0:
        return "0s"
    if ms < 1000:
        return f"{ms}ms"
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs = f"{rest / 1000:.3f}".rstrip("0").rstrip(".")
    if hours:
        return f"{hours}h{minutes}m{secs}s"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def rate_limit_headers(result: LimitResult) -> dict[str, str]:
    limits = result.limits
    headers: dict[str, str] = {}
    if limits.rpm:
        headers["x-ratelimit-limit-requests"] = str(limits.rpm)
        headers["x-ratelimit-remaining-requests"] = str(result.remaining_requests or 0)
        headers["x-ratelimit-reset-requests"] = go_duration(result.reset_requests_s or 0)
    if limits.tpm:
        headers["x-ratelimit-limit-tokens"] = str(limits.tpm)
        headers["x-ratelimit-remaining-tokens"] = str(result.remaining_tokens or 0)
        headers["x-ratelimit-reset-tokens"] = go_duration(result.reset_tokens_s or 0)
    if result.degraded:
        headers["x-gg-ratelimit-degraded"] = "local"
    return headers


def budget_headers(status: BudgetStatus, *, soft_limit_pct: float) -> dict[str, str]:
    headers = {
        "x-gg-budget-limit-usd": usd(status.cap),
        "x-gg-budget-remaining-usd": usd(status.remaining),
        "x-gg-budget-period": status.period,
    }
    if status.ratio >= soft_limit_pct:
        headers["x-gg-budget-warning"] = "soft_limit"
    return headers


def cost_headers(micros: Micros, source: str | None) -> dict[str, str]:
    headers = {"x-gg-cost-usd": usd(micros)}
    if source is not None:
        headers["x-gg-usage-source"] = source
    return headers
