"""the three paid phases (score, generate, judge), each cached on disk and resumable, then grading.

callers inject the generator (gg's own gateway, in process) and the scorer, so this module never talks to a
provider directly and the spend caps of the gateway still apply.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from gg.core.jsonutil import sha256_hex
from gg.core.schema import Message
from gg.routing.base import RoutingRequest, RoutingScorer
from gg.routing.eval.budget import BudgetExceeded, CallEstimate, Estimate, Price, SpendCap, estimate_tokens
from gg.routing.eval.graders import grade
from gg.routing.eval.items import EvalItem, PairConfig, RoleConfig
from gg.routing.eval.judge import ORDERS, JudgePrompt, Order, Verdict, combine, grades_for, parse_verdict
from gg.routing.eval.store import JsonlStore, record_key
from gg.routing.eval.sweep import Outcome

type Role = Literal["weak", "strong"]
ROLES: tuple[Role, ...] = ("weak", "strong")
DEFAULT_JUDGE_TOKENS = 400


class GenerationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    finish_reason: str | None
    served_model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    billed_usd: float
    latency_ms: float


class Generator(Protocol):
    async def generate(
        self, model: str, messages: Sequence[Mapping[str, str]], params: Mapping[str, Any]
    ) -> Generation: ...


type ExtraParams = Callable[[EvalItem, Role], Mapping[str, Any]]
type JudgeExtraParams = Callable[[EvalItem, str, str], Mapping[str, Any]]


@dataclass(slots=True)
class Stores:
    scores: JsonlStore
    generations: JsonlStore
    judgements: JsonlStore

    @classmethod
    def open(cls, directory: Path) -> "Stores":
        return cls(
            JsonlStore(directory / "scores.jsonl"),
            JsonlStore(directory / "generations.jsonl"),
            JsonlStore(directory / "judgements.jsonl"),
        )


@dataclass(slots=True)
class PhaseStats:
    cached: int = 0
    called: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=lambda: [])

    def to_json(self) -> dict[str, Any]:
        return {
            "cached": self.cached,
            "called": self.called,
            "failed": self.failed,
            "errors": self.errors[:20],
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RoutingEvalRun:
    def __init__(
        self,
        items: Sequence[EvalItem],
        pair: PairConfig,
        *,
        stores: Stores,
        generator: Generator,
        scorer: RoutingScorer,
        prices: Mapping[str, Price],
        judge_prompt: JudgePrompt,
        cap: SpendCap,
        scorer_cost_usd: float = 0.0,
        concurrency: int = 4,
        extra_params: ExtraParams | None = None,
        judge_extra: JudgeExtraParams | None = None,
    ) -> None:
        self.items = list(items)
        self.pair = pair
        self.stores = stores
        self.generator = generator
        self.scorer = scorer
        self.prices = prices
        self.judge_prompt = judge_prompt
        self.cap = cap
        self.scorer_cost_usd = scorer_cost_usd
        self._semaphore = asyncio.Semaphore(concurrency)
        self._extra = extra_params
        self._judge_extra = judge_extra
        self.stats = {"score": PhaseStats(), "generate": PhaseStats(), "judge": PhaseStats()}
        self.budget_stop: str | None = None

    # keys

    def role(self, role: Role) -> RoleConfig:
        return self.pair.weak if role == "weak" else self.pair.strong

    def score_key(self, item: EvalItem) -> str:
        return record_key("score", self.scorer.version, item.prompt_sha)

    def gen_params(self, item: EvalItem, role: Role) -> dict[str, Any]:
        cfg = self.role(role)
        params: dict[str, Any] = {**cfg.params, "max_completion_tokens": item.max_tokens + cfg.extra_tokens}
        if self._extra is not None:
            params.update(self._extra(item, role))
        return params

    def gen_key(self, item: EvalItem, role: Role) -> str:
        return record_key("generation", item.prompt_sha, self.role(role).model, self.gen_params(item, role))

    def judge_params(self, item: EvalItem, answer_a: str, answer_b: str) -> dict[str, Any]:
        cfg = self.pair.judge
        params: dict[str, Any] = {
            **cfg.params,
            "max_completion_tokens": cfg.max_tokens or DEFAULT_JUDGE_TOKENS,
        }
        if self._judge_extra is not None:
            params.update(self._judge_extra(item, answer_a, answer_b))
        return params

    def _answers(self, item: EvalItem, order: Order) -> tuple[str, str] | None:
        weak = self.stores.generations.get(self.gen_key(item, "weak"))
        strong = self.stores.generations.get(self.gen_key(item, "strong"))
        if weak is None or strong is None:
            return None
        return (weak["text"], strong["text"]) if order == "ws" else (strong["text"], weak["text"])

    def judge_key(self, item: EvalItem, order: Order, answer_a: str, answer_b: str) -> str:
        return record_key(
            "judgement",
            item.prompt_sha,
            sha256_hex(answer_a),
            sha256_hex(answer_b),
            self.pair.judge.model,
            self.judge_params(item, answer_a, answer_b),
            self.judge_prompt.version,
            order,
        )

    # estimate

    def price(self, model: str) -> Price:
        price = self.prices.get(model)
        if price is None:
            raise ValueError(f"no price for model {model}")
        return price

    def _gen_estimate(self, item: EvalItem, role: Role) -> CallEstimate:
        model = self.role(role).model
        tokens = estimate_tokens(item.prompt_chars, len(item.messages))
        return CallEstimate(model, tokens, item.max_tokens + self.role(role).extra_tokens, self.price(model))

    def _judge_estimate(self, item: EvalItem, answers: tuple[str, str] | None) -> CallEstimate:
        if answers is None:
            answer_tokens = 2 * item.max_tokens
            chars = len(self.judge_prompt.template) + item.prompt_chars + len(item.reference or "")
            tokens = estimate_tokens(chars) + answer_tokens
        else:
            tokens = estimate_tokens(len(self.judge_prompt.render(item, *answers)))
        model = self.pair.judge.model
        return CallEstimate(
            model, tokens, self.pair.judge.max_tokens or DEFAULT_JUDGE_TOKENS, self.price(model)
        )

    def estimate(self, phases: Iterable[str]) -> Estimate:
        wanted = set(phases)
        calls: list[CallEstimate] = []
        scores = 0
        for item in self.items:
            if "score" in wanted and self.score_key(item) not in self.stores.scores:
                scores += 1
            if "generate" in wanted:
                calls += [
                    self._gen_estimate(item, role)
                    for role in ROLES
                    if self.gen_key(item, role) not in self.stores.generations
                ]
            if "judge" in wanted and item.needs_judge:
                for order in ORDERS:
                    answers = self._answers(item, order)
                    if answers is None or self.judge_key(item, order, *answers) not in self.stores.judgements:
                        calls.append(self._judge_estimate(item, answers))
        return Estimate.of(calls, extra_billed_usd=scores * self.scorer_cost_usd)

    # phases

    async def _guarded[T](self, phase: str, worst_usd: float, call: Callable[[], Awaitable[T]]) -> T | None:
        stats = self.stats[phase]
        if self.budget_stop is not None:
            return None
        async with self._semaphore:
            try:
                held = self.cap.reserve(worst_usd)
            except BudgetExceeded as exc:
                self.budget_stop = str(exc)
                return None
            actual = 0.0
            try:
                result = await call()
            except GenerationError as exc:
                stats.failed += 1
                stats.errors.append(str(exc))
                return None
            else:
                actual = getattr(result, "billed_usd", 0.0)
                stats.called += 1
                return result
            finally:
                self.cap.settle(held, actual)

    async def score_all(self) -> None:
        stats = self.stats["score"]

        async def one(item: EvalItem) -> None:
            key = self.score_key(item)
            if key in self.stores.scores:
                stats.cached += 1
                return

            async def call() -> _Billed:
                messages = tuple(Message.model_validate(m) for m in item.wire_messages())
                req = RoutingRequest(
                    request_id=f"eval-{item.id}",
                    key_id="routing-eval",
                    messages=messages,
                    tools_present=False,
                )
                score = await self.scorer.score(req)
                if score.fallback:
                    raise GenerationError(f"{item.id}: scorer fallback ({score.fallback_reason})")
                self.stores.scores.put(
                    key,
                    {
                        "item_id": item.id,
                        "scorer": score.scorer,
                        "scorer_version": score.scorer_version,
                        "score": score.score,
                        "raw_score": score.raw_score,
                        "tier": score.tier,
                        "latency_ms": round(score.latency_ms, 1),
                        "cost_usd": score.cost_usd or 0.0,
                        # local analysis only; never training data (typesafe mca 2.3(b))
                        "raw": dict(score.raw),
                        "created_at": _now(),
                    },
                )
                return _Billed(score.cost_usd or 0.0)

            await self._guarded("score", self.scorer_cost_usd, call)

        await asyncio.gather(*(one(item) for item in self.items))

    async def generate_all(self) -> None:
        stats = self.stats["generate"]

        async def one(item: EvalItem, role: Role) -> None:
            key = self.gen_key(item, role)
            if key in self.stores.generations:
                stats.cached += 1
                return
            model = self.role(role).model
            params = self.gen_params(item, role)

            async def call() -> Generation:
                gen = await self.generator.generate(model, item.wire_messages(), params)
                if gen.served_model != model:
                    # a fallback answer is contaminated; never mix it in (C11 §3.8)
                    raise GenerationError(f"{item.id}/{role}: served by {gen.served_model}, not {model}")
                self.stores.generations.put(
                    key,
                    {
                        "item_id": item.id,
                        "role": role,
                        "model": model,
                        "params": params,
                        "text": gen.text,
                        "finish_reason": gen.finish_reason,
                        "input_tokens": gen.input_tokens,
                        "output_tokens": gen.output_tokens,
                        "reasoning_tokens": gen.reasoning_tokens,
                        "cost_usd": gen.cost_usd,
                        "billed_usd": gen.billed_usd,
                        "latency_ms": round(gen.latency_ms, 1),
                        "created_at": _now(),
                    },
                )
                return gen

            await self._guarded("generate", self._gen_estimate(item, role).billed_worst_usd, call)

        await asyncio.gather(*(one(item, role) for item in self.items for role in ROLES))

    async def judge_all(self) -> None:
        stats = self.stats["judge"]

        async def one(item: EvalItem, order: Order) -> None:
            answers = self._answers(item, order)
            if answers is None:
                return
            key = self.judge_key(item, order, *answers)
            if key in self.stores.judgements:
                stats.cached += 1
                return
            prompt = self.judge_prompt.render(item, *answers)
            params = self.judge_params(item, *answers)
            model = self.pair.judge.model

            async def call() -> Generation:
                gen = await self.generator.generate(model, [{"role": "user", "content": prompt}], params)
                self.stores.judgements.put(
                    key,
                    {
                        "item_id": item.id,
                        "order": order,
                        "judge": model,
                        "prompt_version": self.judge_prompt.version,
                        "verdict": parse_verdict(gen.text),
                        "text": gen.text,
                        "cost_usd": gen.cost_usd,
                        "billed_usd": gen.billed_usd,
                        "created_at": _now(),
                    },
                )
                return gen

            await self._guarded("judge", self._judge_estimate(item, answers).billed_worst_usd, call)

        await asyncio.gather(
            *(one(item, order) for item in self.items if item.needs_judge for order in ORDERS)
        )

    async def run(self, phases: Iterable[str]) -> None:
        wanted = set(phases)
        if "score" in wanted:
            await self.score_all()
        if "generate" in wanted:
            await self.generate_all()
        if "judge" in wanted:
            await self.judge_all()

    # grading

    def collect(self) -> "Collected":
        outcomes: list[Outcome] = []
        scores: dict[str, float] = {}
        missing: dict[str, list[str]] = {"score": [], "generation": [], "judgement": []}
        judge_pairs = 0
        judge_consistent = 0
        judge_invalid = 0
        scorer_costs: list[float] = []
        for item in self.items:
            score = self.stores.scores.get(self.score_key(item))
            weak = self.stores.generations.get(self.gen_key(item, "weak"))
            strong = self.stores.generations.get(self.gen_key(item, "strong"))
            if score is None:
                missing["score"].append(item.id)
            if weak is None or strong is None:
                missing["generation"].append(item.id)
                continue
            if item.needs_judge:
                verdicts: dict[Order, Verdict] = {}
                for order in ORDERS:
                    answers = self._answers(item, order)
                    record = (
                        None
                        if answers is None
                        else self.stores.judgements.get(self.judge_key(item, order, *answers))
                    )
                    if record is not None:
                        verdicts[order] = record["verdict"]
                if len(verdicts) < len(ORDERS):
                    missing["judgement"].append(item.id)
                    continue
                winner, agreed = combine(verdicts["ws"], verdicts["sw"])
                judge_pairs += 1
                judge_consistent += agreed
                judge_invalid += sum(v == "invalid" for v in verdicts.values())
                g_weak, g_strong = grades_for(winner)
            else:
                g_weak = grade(item.grader, weak["text"]) or 0.0
                g_strong = grade(item.grader, strong["text"]) or 0.0
            if score is None:
                continue
            scores[item.id] = float(score["score"])
            scorer_costs.append(float(score.get("cost_usd") or 0.0))
            outcomes.append(
                Outcome(
                    item_id=item.id,
                    category=item.category,
                    split=item.split,
                    g_weak=g_weak,
                    g_strong=g_strong,
                    cost_weak=float(weak["cost_usd"]),
                    cost_strong=float(strong["cost_usd"]),
                )
            )
        scorer_cost = sum(scorer_costs) / len(scorer_costs) if scorer_costs else 0.0
        return Collected(
            outcomes=outcomes,
            scores=scores,
            missing=missing,
            judge_pairs=judge_pairs,
            judge_consistent=judge_consistent,
            judge_invalid=judge_invalid,
            scorer_cost_usd=scorer_cost,
        )


@dataclass(frozen=True, slots=True)
class _Billed:
    billed_usd: float


@dataclass(frozen=True, slots=True)
class Collected:
    outcomes: list[Outcome]
    scores: dict[str, float]
    missing: dict[str, list[str]]
    judge_pairs: int
    judge_consistent: int
    judge_invalid: int
    scorer_cost_usd: float
