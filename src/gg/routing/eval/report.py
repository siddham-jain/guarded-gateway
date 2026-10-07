"""metrics, baselines and the report artefacts: result json, markdown, curve csv and a dependency-free svg"""

import math
import platform
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from html import escape
from typing import Any

from gg.routing.eval.items import EvalItem
from gg.routing.eval.run import Collected
from gg.routing.eval.scorers import heuristic_score
from gg.routing.eval.sweep import (
    CurvePoint,
    Outcome,
    alpha_for_pgr,
    apgr_bins,
    apgr_trapz,
    at_alpha,
    auroc,
    bootstrap_ci,
    cpt,
    curve,
    endpoints,
    interpolate,
    oracle_scores,
    pgr,
    random_curve,
    wilson_upper,
)

SUITE = "routing"
PRESETS = {"economy": 0.5, "balanced": 0.8, "quality": 0.95}
SWEEP_ALPHAS = tuple(round(0.05 * i, 2) for i in range(21))
RANDOM_SEED = 1337


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or math.isinf(value) else round(value, digits)


def _router_scores(collected: Collected, items: Mapping[str, EvalItem]) -> dict[str, dict[str, float]]:
    outcomes = collected.outcomes
    rng = random.Random(RANDOM_SEED)  # noqa: S311
    return {
        "scorer": collected.scores,
        "heuristic": {o.item_id: heuristic_score(items[o.item_id]) for o in outcomes},
        "random": {o.item_id: rng.random() for o in outcomes},
        "oracle": oracle_scores(outcomes),
    }


def _points(
    name: str, outcomes: Sequence[Outcome], scores: Mapping[str, float], extra: float
) -> list[CurvePoint]:
    if name == "random":
        # the analytic expectation, not one noisy draw
        return random_curve(outcomes)
    return curve(outcomes, scores, extra)


def _router_metrics(
    name: str, outcomes: Sequence[Outcome], scores: Mapping[str, float], extra: float, resamples: int
) -> dict[str, Any]:
    if not outcomes:
        return {}
    q_weak, q_strong = endpoints(outcomes)
    points = _points(name, outcomes, scores, extra)
    labels = [o.strong_wins for o in outcomes]

    def stat(sample: Sequence[Outcome]) -> float | None:
        w, s = endpoints(sample)
        sample_scores = oracle_scores(sample) if name == "oracle" else scores
        return apgr_trapz(_points(name, sample, sample_scores, extra), w, s)

    ci = None if name == "random" or resamples <= 0 else bootstrap_ci(outcomes, stat, resamples=resamples)
    return {
        "apgr": _r(apgr_trapz(points, q_weak, q_strong)),
        "apgr_ci95": None if ci is None else [_r(ci[0]), _r(ci[1])],
        "apgr_10bin": _r(apgr_bins(points, q_weak, q_strong)),
        "cpt50": _r(cpt(points, 0.5, q_weak, q_strong)),
        "cpt80": _r(cpt(points, 0.8, q_weak, q_strong)),
        "auroc": None if name == "random" else _r(auroc([scores[o.item_id] for o in outcomes], labels)),
    }


def _endpoint_block(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    q_weak, q_strong = endpoints(outcomes)
    n = len(outcomes)
    return {
        "n": n,
        "quality_weak": _r(q_weak),
        "quality_strong": _r(q_strong),
        "cost_per_1k_weak": _r(1000 * sum(o.cost_weak for o in outcomes) / n) if n else None,
        "cost_per_1k_strong": _r(1000 * sum(o.cost_strong for o in outcomes) / n) if n else None,
        "strong_wins": sum(o.strong_wins for o in outcomes),
        "weak_wins": sum(o.g_weak > o.g_strong for o in outcomes),
    }


def _operating_point(
    outcomes: Sequence[Outcome], scores: Mapping[str, float], alpha: float, extra: float
) -> dict[str, Any]:
    point = at_alpha(outcomes, scores, alpha, extra)
    q_weak, q_strong = endpoints(outcomes)
    random_quality, random_cost = interpolate(random_curve(outcomes), point.strong_share)
    cost_strong = sum(o.cost_strong for o in outcomes) / len(outcomes)
    return {
        "alpha": _r(alpha),
        "strong_share": _r(point.strong_share),
        "quality": _r(point.quality),
        "pgr": _r(point.pgr),
        "cost_per_1k": _r(1000 * point.cost),
        "saved_vs_strong_pct": _r(100 * (1 - point.cost / cost_strong)) if cost_strong > 0 else None,
        "under_routed": point.under_routed,
        "strong_wins": point.strong_wins,
        "under_route_rate": _r(point.under_route_rate),
        "under_route_upper95": _r(wilson_upper(point.under_routed, point.strong_wins))
        if point.strong_wins
        else None,
        "random_same_share": {
            "quality": _r(random_quality),
            "pgr": _r(pgr(random_quality, q_weak, q_strong)),
            "cost_per_1k": _r(1000 * random_cost),
        },
    }


def build_result(
    collected: Collected,
    items: Sequence[EvalItem],
    *,
    meta: Mapping[str, Any],
    current_alpha: float,
    resamples: int = 1000,
) -> dict[str, Any]:
    by_id = {i.id: i for i in items}
    outcomes = collected.outcomes
    routers = _router_scores(collected, by_id)
    extra = {"scorer": collected.scorer_cost_usd}
    splits = {
        "all": outcomes,
        "tune": [o for o in outcomes if o.split == "tune"],
        "heldout": [o for o in outcomes if o.split == "heldout"],
    }
    metrics = {
        split: {
            "endpoints": _endpoint_block(subset),
            "routers": {
                name: _router_metrics(name, subset, scores, extra.get(name, 0.0), resamples)
                for name, scores in routers.items()
            },
        }
        for split, subset in splits.items()
        if subset
    }
    scorer_scores = routers["scorer"]
    scorer_extra = extra["scorer"]
    presets: dict[str, Any] = {}
    tune = splits["tune"]
    if tune:
        w, s = endpoints(tune)
        tune_points = curve(tune, scorer_scores, scorer_extra)
        for preset, target in PRESETS.items():
            alpha = alpha_for_pgr(tune_points, target, w, s)
            presets[preset] = {
                "target_pgr": target,
                "alpha": _r(alpha),
                "heldout": None
                if alpha is None or not splits["heldout"]
                else _operating_point(splits["heldout"], scorer_scores, alpha, scorer_extra),
            }
    sweep = (
        [_operating_point(outcomes, scorer_scores, alpha, scorer_extra) for alpha in SWEEP_ALPHAS]
        if outcomes
        else []
    )
    categories: dict[str, Any] = {}
    for category in sorted({o.category for o in outcomes}):
        subset = [o for o in outcomes if o.category == category]
        block = _endpoint_block(subset)
        point = at_alpha(subset, scorer_scores, current_alpha)
        block["strong_share_at_current_alpha"] = _r(point.strong_share)
        block["quality_at_current_alpha"] = _r(point.quality)
        categories[category] = block
    prior_agreement = [
        (scorer_scores[o.item_id] >= current_alpha) == (by_id[o.item_id].expected_tier == "strong")
        for o in outcomes
    ]
    curves = (
        {
            name: [
                {
                    "strong_share": _r(p.strong_share),
                    "quality": _r(p.quality),
                    "cost_per_1k": _r(1000 * p.cost),
                    "alpha": _r(p.alpha),
                }
                for p in _points(name, outcomes, scores, extra.get(name, 0.0))
            ]
            for name, scores in routers.items()
        }
        if outcomes
        else {}
    )
    return {
        "schema_version": 1,
        "suite": SUITE,
        "run_id": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
        **meta,
        "env": {"python": platform.python_version()},
        "dataset": {
            **meta.get("dataset", {}),
            "evaluated": len(outcomes),
            "splits": {k: len(v) for k, v in splits.items()},
        },
        "missing": {k: v for k, v in collected.missing.items() if v},
        "judge": {
            "pairs": collected.judge_pairs,
            "position_consistency": _r(collected.judge_consistent / collected.judge_pairs)
            if collected.judge_pairs
            else None,
            "invalid_verdicts": collected.judge_invalid,
        },
        "scorer_cost_per_1k": _r(1000 * collected.scorer_cost_usd),
        "current_alpha": current_alpha,
        "at_current_alpha": _operating_point(outcomes, scorer_scores, current_alpha, scorer_extra)
        if outcomes
        else None,
        "prior_agreement_at_current_alpha": _r(sum(prior_agreement) / len(prior_agreement))
        if prior_agreement
        else None,
        "metrics": metrics,
        "presets": presets,
        "sweep": sweep,
        "by_category": categories,
        "curves": curves,
    }


def _fmt(value: Any, pct: bool = False, money: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{100 * value:.1f}%"
    if money:
        return f"${value:.4f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _ci(value: Any) -> str:
    return "—" if not value else f"[{value[0]:.2f}, {value[1]:.2f}]"


ROUTER_LABELS = {
    "scorer": "scorer (GG)",
    "heuristic": "length/keyword heuristic",
    "random": "random",
    "oracle": "oracle",
}


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[str]) -> list[str]:
    return [_row(*header), _row(*(["---"] * len(header))), *rows, ""]


def _setup(result: Mapping[str, Any]) -> list[str]:
    pair = result["pair"]
    data = result["dataset"]
    splits = data["splits"]
    lines = [
        f"# Routing eval - {result['mode']} - {result['run_id']}",
        "",
        f"Pair `{pair['name']}` (profile `{pair['profile']}`): weak `{pair['weak']}`, "
        f"strong `{pair['strong']}`, judge `{pair['judge']}` (prompt `{result['judge_prompt']}`). "
        f"Scorer `{result['scorer_version']}`.",
        f"Dataset `{data['sha256'][:12]}`: {data['n_items']} items, {data['evaluated']} evaluated "
        f"(tune {splits.get('tune', 0)}, held-out {splits.get('heldout', 0)}).",
        "",
    ]
    if result["missing"]:
        gaps = ", ".join(f"{k} {len(v)}" for k, v in result["missing"].items())
        lines += [f"Missing: {gaps} (rerun to fill; missing items are left out of every metric).", ""]
    return lines


def _routers(result: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for split in ("all", "heldout"):
        block = result["metrics"].get(split)
        if block is None:
            continue
        ep = block["endpoints"]
        weak = f"{_fmt(ep['quality_weak'])} at {_fmt(ep['cost_per_1k_weak'], money=True)}/1k"
        strong = f"{_fmt(ep['quality_strong'])} at {_fmt(ep['cost_per_1k_strong'], money=True)}/1k"
        lines += [
            f"## Routers - {split} (n={ep['n']})",
            "",
            f"always-weak quality {weak} · always-strong quality {strong} · "
            f"strong wins {ep['strong_wins']}, weak wins {ep['weak_wins']}",
            "",
        ]
        rows = [
            _row(
                ROUTER_LABELS[name],
                _fmt(m.get("apgr")),
                _ci(m.get("apgr_ci95")),
                _fmt(m.get("apgr_10bin")),
                _fmt(m.get("cpt50"), pct=True),
                _fmt(m.get("cpt80"), pct=True),
                _fmt(m.get("auroc")),
            )
            for name, m in block["routers"].items()
        ]
        lines += _table(("Router", "APGR", "95% CI", "APGR (10-bin)", "CPT(50%)", "CPT(80%)", "AUROC"), rows)
    return lines


def _current(result: Mapping[str, Any]) -> list[str]:
    c = result["at_current_alpha"]
    if not c:
        return []
    rnd = c["random_same_share"]
    under = f"{c['under_routed']}/{c['strong_wins']} (95% upper {_fmt(c['under_route_upper95'], pct=True)})"
    prior = _fmt(result["prior_agreement_at_current_alpha"], pct=True)
    return [
        f"## At the configured alpha = {result['current_alpha']}",
        "",
        f"{_fmt(c['strong_share'], pct=True)} strong · quality {_fmt(c['quality'])} · "
        f"PGR {_fmt(c['pgr'])} · {_fmt(c['cost_per_1k'], money=True)}/1k "
        f"(saves {_pct_value(c['saved_vs_strong_pct'])} vs always-strong) · under-routed {under} · "
        f"random at the same share: quality {_fmt(rnd['quality'])}, PGR {_fmt(rnd['pgr'])}",
        "",
        f"Agreement with the authors' tier prior: {prior}.",
        "",
    ]


def _presets(result: Mapping[str, Any]) -> list[str]:
    if not result["presets"]:
        return []
    rows: list[str] = []
    for name, p in result["presets"].items():
        h = p["heldout"] or {}
        rows.append(
            _row(
                name,
                p["target_pgr"],
                _fmt(p["alpha"]),
                _fmt(h.get("strong_share"), pct=True),
                _fmt(h.get("quality")),
                _fmt(h.get("pgr")),
                _fmt(h.get("cost_per_1k"), money=True),
                _pct_value(h.get("saved_vs_strong_pct")),
                f"{h.get('under_routed', '—')}/{h.get('strong_wins', '—')}",
            )
        )
    header = (
        "Preset",
        "target PGR",
        "alpha",
        "% strong",
        "quality",
        "PGR",
        "$/1k",
        "saved vs strong",
        "under",
    )
    return ["## Presets (alpha chosen on tune, reported on held-out)", "", *_table(header, rows)]


def _sweep(result: Mapping[str, Any]) -> list[str]:
    rows = [
        _row(
            f"{row['alpha']:.2f}",
            _fmt(row["strong_share"], pct=True),
            _fmt(row["quality"]),
            _fmt(row["pgr"]),
            _fmt(row["cost_per_1k"], money=True),
            _fmt(row["random_same_share"]["pgr"]),
        )
        for row in result["sweep"]
    ]
    header = ("alpha", "% strong", "quality", "PGR", "$/1k", "random PGR at same share")
    return ["## Alpha sweep (scorer, all items)", "", *_table(header, rows)]


def _categories(result: Mapping[str, Any]) -> list[str]:
    rows = [
        _row(
            name,
            c["n"],
            _fmt(c["quality_weak"]),
            _fmt(c["quality_strong"]),
            c["strong_wins"],
            _fmt(c["strong_share_at_current_alpha"], pct=True),
            _fmt(c["quality_at_current_alpha"]),
        )
        for name, c in result["by_category"].items()
    ]
    header = ("Category", "n", "weak q", "strong q", "strong wins", "% strong at alpha", "q at alpha")
    return ["## By category", "", *_table(header, rows)]


def _spend(result: Mapping[str, Any]) -> list[str]:
    judge = result["judge"]
    cost = result["cost"]
    phases = ", ".join(
        f"{k} {v['called']} called / {v['cached']} cached / {v['failed']} failed"
        for k, v in cost["phases"].items()
    )
    stop = f" Stopped: {cost['budget_stop']}" if cost.get("budget_stop") else ""
    consistency = _fmt(judge["position_consistency"], pct=True)
    return [
        "## Judge and spend",
        "",
        f"Judge pairs {judge['pairs']}, position consistency {consistency}, invalid verdicts "
        f"{judge['invalid_verdicts']}. Scorer cost {_fmt(result['scorer_cost_per_1k'], money=True)}/1k.",
        f"This invocation billed ${cost['spent_usd']:.4f} of a ${cost['cap_usd']:.2f} cap; {phases}.{stop}",
        "",
        "Cost-quality curve: `cost_quality.svg` (data in `curves.csv`).",
        "",
    ]


def render_markdown(result: Mapping[str, Any]) -> str:
    sections = (_setup, _routers, _current, _presets, _sweep, _categories, _spend)
    return "\n".join(line for section in sections for line in section(result))


def render_csv(result: Mapping[str, Any]) -> str:
    rows = ["router,strong_share,quality,cost_per_1k,alpha"]
    for name, points in result["curves"].items():
        for p in points:
            alpha = "" if p["alpha"] is None else p["alpha"]
            rows.append(f"{name},{p['strong_share']},{p['quality']},{p['cost_per_1k']},{alpha}")
    return "\n".join(rows) + "\n"


# categorical slots 1-4 of the reference palette, in fixed order; random also gets a dash
SERIES = {"scorer": "#2a78d6", "heuristic": "#eb6834", "oracle": "#1baf7a", "random": "#52514e"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
WIDTH, HEIGHT = 720, 440
LEFT, RIGHT, TOP, BOTTOM = 64, 180, 40, 56


def _svg_open(height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="system-ui, sans-serif" font-size="12">'
    )


def render_svg(result: Mapping[str, Any]) -> str:
    curves: Mapping[str, list[dict[str, Any]]] = result["curves"]
    all_points = [p for pts in curves.values() for p in pts]
    if not all_points:
        return _svg_open(60) + '<text x="10" y="30">no data</text></svg>\n'
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    x_max = max(p["cost_per_1k"] for p in all_points) or 1.0
    q_values = [p["quality"] for p in all_points]
    y_min = max(0.0, math.floor(min(q_values) * 10) / 10)
    y_max = min(1.0, math.ceil(max(q_values) * 10) / 10)
    if y_max <= y_min:
        y_max = y_min + 0.1

    def sx(v: float) -> float:
        return LEFT + plot_w * v / x_max

    def sy(v: float) -> float:
        return TOP + plot_h * (1 - (v - y_min) / (y_max - y_min))

    title = f"Cost-quality curve ({escape(result['pair']['name'])}, {escape(result['mode'])})"
    out = [
        _svg_open(HEIGHT),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
        f'<text x="{LEFT}" y="22" font-size="14" fill="{INK}">{title}</text>',
    ]
    base = TOP + plot_h
    for i in range(6):
        yv, xv = y_min + (y_max - y_min) * i / 5, x_max * i / 5
        out += [
            f'<line x1="{LEFT}" x2="{LEFT + plot_w}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="{GRID}"/>',
            f'<text x="{LEFT - 8}" y="{sy(yv) + 4:.1f}" text-anchor="end" fill="{MUTED}">{yv:.2f}</text>',
            f'<text x="{sx(xv):.1f}" y="{base + 18}" text-anchor="middle" fill="{MUTED}">${xv:.3f}</text>',
        ]
    out += [
        f'<line x1="{LEFT}" x2="{LEFT + plot_w}" y1="{base}" y2="{base}" stroke="{MUTED}"/>',
        f'<text x="{LEFT + plot_w / 2}" y="{HEIGHT - 12}" text-anchor="middle" fill="{INK}">'
        "cost per 1k requests (USD, list price)</text>",
        f'<text transform="translate(16 {TOP + plot_h / 2}) rotate(-90)" text-anchor="middle" fill="{INK}">'
        "quality (mean grade)</text>",
    ]
    for index, (name, points) in enumerate(curves.items()):
        colour = SERIES.get(name, "#4a3aa7")
        label = escape(ROUTER_LABELS.get(name, name))
        dash = ' stroke-dasharray="6 4"' if name == "random" else ""
        coords = " ".join(f"{sx(p['cost_per_1k']):.1f},{sy(p['quality']):.1f}" for p in points)
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="{colour}" stroke-width="2"{dash} '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        step = max(1, len(points) // 60)
        for p in points[::step]:
            alpha = "" if p["alpha"] is None else f", alpha={p['alpha']}"
            tip = (
                f"{label}: {100 * p['strong_share']:.0f}% strong, quality {p['quality']:.3f}, "
                f"${p['cost_per_1k']:.4f}/1k{alpha}"
            )
            out.append(
                f'<circle cx="{sx(p["cost_per_1k"]):.1f}" cy="{sy(p["quality"]):.1f}" r="4" fill="{colour}" '
                f'stroke="{SURFACE}" stroke-width="2"><title>{tip}</title></circle>'
            )
        ly, lx = TOP + 16 + 22 * index, LEFT + plot_w + 16
        out += [
            f'<line x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}" stroke="{colour}" stroke-width="2"{dash}/>',
            f'<text x="{lx + 28}" y="{ly + 4}" fill="{INK}">{label}</text>',
        ]
    out.append("</svg>")
    return "\n".join(out) + "\n"
