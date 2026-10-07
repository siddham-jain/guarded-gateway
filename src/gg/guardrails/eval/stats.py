import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% wilson score interval; stays inside [0, 1] and is honest at 0/n and n/n"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int, *, direction: str) -> dict[str, object]:
    low, high = wilson(k, n)
    value = k / n if n else 0.0
    return {
        "value": round(value, 4),
        "n": n,
        "k": k,
        "ci95": [round(low, 4), round(high, 4)],
        "direction": direction,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]
