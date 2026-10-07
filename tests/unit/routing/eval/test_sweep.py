import math

import pytest

from gg.routing.eval.sweep import (
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


def outcome(
    item_id: str, g_weak: float, g_strong: float, cost_weak: float = 1.0, cost_strong: float = 10.0
) -> Outcome:
    return Outcome(item_id, "cat", "tune", g_weak, g_strong, cost_weak, cost_strong)


# four items: strong helps on a and b only
TOY = [outcome("a", 0, 1), outcome("b", 0, 1), outcome("c", 1, 1), outcome("d", 0, 0)]


def test_endpoints_and_pgr() -> None:
    assert endpoints(TOY) == (0.25, 0.75)
    assert pgr(0.5, 0.25, 0.75) == 0.5
    assert pgr(0.5, 0.75, 0.75) is None


def test_curve_walks_score_order_with_costs() -> None:
    points = curve(TOY, {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    assert [(p.strong_share, p.quality, p.cost) for p in points] == [
        (0.0, 0.25, 1.0),
        (0.25, 0.5, 3.25),
        (0.5, 0.75, 5.5),
        (0.75, 0.75, 7.75),
        (1.0, 0.75, 10.0),
    ]
    assert points[0].alpha == math.inf
    assert [p.alpha for p in points[1:]] == [0.9, 0.8, 0.2, 0.1]


def test_perfect_router_matches_oracle_and_reversed_router_is_worse_than_random() -> None:
    w, s = endpoints(TOY)
    perfect = apgr_trapz(curve(TOY, {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}), w, s)
    oracle = apgr_trapz(curve(TOY, oracle_scores(TOY)), w, s)
    reversed_ = apgr_trapz(curve(TOY, {"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9}), w, s)
    # trapz: 0.25*(0.25+0.5)/2 + 0.25*(0.5+0.75)/2 + 0.5*0.75 = 0.625 -> (0.625-0.25)/0.5
    assert perfect == pytest.approx(0.75)
    assert oracle == pytest.approx(perfect)
    assert reversed_ is not None
    assert reversed_ < 0.5


def test_random_router_is_one_half_analytically() -> None:
    w, s = endpoints(TOY)
    line = random_curve(TOY)
    assert apgr_trapz(line, w, s) == pytest.approx(0.5)
    assert cpt(line, 0.5, w, s) == pytest.approx(0.5)
    assert cpt(line, 0.8, w, s) == pytest.approx(0.8)
    assert apgr_bins(line, w, s) == pytest.approx(0.55)


def test_one_tie_block_is_the_random_line() -> None:
    w, s = endpoints(TOY)
    points = curve(TOY, dict.fromkeys("abcd", 0.5))
    assert [p.strong_share for p in points] == [0.0, 1.0]
    assert apgr_trapz(points, w, s) == pytest.approx(0.5)


def test_tie_block_is_interpolated() -> None:
    points = curve(TOY, {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1})
    # b and c tie: halfway through their block quality is the average of the two block ends
    quality, cost = interpolate(points, 0.5)
    assert quality == pytest.approx((0.5 + 0.75) / 2)
    assert cost == pytest.approx((3.25 + 7.75) / 2)


def test_cpt_interpolates_inside_a_segment() -> None:
    w, s = endpoints(TOY)
    points = curve(TOY, {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    assert cpt(points, 0.5, w, s) == pytest.approx(0.25)
    assert cpt(points, 0.75, w, s) == pytest.approx(0.375)
    assert cpt(points, 1.1, w, s) is None


def test_apgr_bins_on_the_perfect_router() -> None:
    w, s = endpoints(TOY)
    points = curve(TOY, {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    # pgr climbs 2 per unit share up to 1 at 50% strong: bins 0.1..0.4 give 0.2, 0.4, 0.6, 0.8
    expected = (0.2 + 0.4 + 0.6 + 0.8 + 1.0 * 6) / 10
    assert apgr_bins(points, w, s) == pytest.approx(expected)


def test_apgr_is_undefined_without_a_gap() -> None:
    flat = [outcome("a", 1, 1), outcome("b", 0, 0)]
    w, s = endpoints(flat)
    assert apgr_trapz(curve(flat, {"a": 1, "b": 0}), w, s) is None
    assert cpt(curve(flat, {"a": 1, "b": 0}), 0.5, w, s) is None


def test_at_alpha_counts_under_routing() -> None:
    point = at_alpha(TOY, {"a": 0.9, "b": 0.3, "c": 0.6, "d": 0.1}, 0.5, extra_cost=0.5)
    assert point.strong_share == 0.5
    assert point.quality == 0.5
    assert point.pgr == pytest.approx(0.5)
    assert point.cost == pytest.approx((10 + 1 + 10 + 1) / 4 + 0.5)
    assert (point.under_routed, point.strong_wins) == (1, 2)
    assert point.under_route_rate == 0.5


def test_alpha_for_pgr_picks_the_cheapest_point() -> None:
    w, s = endpoints(TOY)
    points = curve(TOY, {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    assert alpha_for_pgr(points, 0.5, w, s) == 0.9
    assert alpha_for_pgr(points, 0.8, w, s) == 0.8
    assert alpha_for_pgr(points, 1.5, w, s) is None


def test_auroc_reference_values() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    # sklearn.metrics.roc_auc_score([1, 0, 1, 0, 1], [0.8, 0.4, 0.4, 0.2, 0.9]) == 0.9166...
    assert auroc([0.8, 0.4, 0.4, 0.2, 0.9], [True, False, True, False, True]) == pytest.approx(11 / 12)
    assert auroc([0.1, 0.2], [True, True]) is None


def test_wilson_upper_is_honest_at_zero() -> None:
    assert wilson_upper(0, 150) == pytest.approx(0.0249, abs=1e-3)
    assert wilson_upper(0, 0) == 1.0


def test_bootstrap_is_reproducible_with_a_seed() -> None:
    outcomes = [outcome(f"i{n}", n % 3 == 0, n % 2 == 0) for n in range(40)]
    scores = {o.item_id: (n * 7 % 11) / 10 for n, o in enumerate(outcomes)}

    def stat(sample: list[Outcome]) -> float | None:
        w, s = endpoints(sample)
        return apgr_trapz(curve(sample, scores), w, s)

    first = bootstrap_ci(outcomes, stat, resamples=200, seed=3)
    assert first is not None
    assert first == bootstrap_ci(outcomes, stat, resamples=200, seed=3)
    assert first[0] <= first[1]
