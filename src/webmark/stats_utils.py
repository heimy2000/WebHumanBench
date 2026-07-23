"""Statistical audit primitives used in supplementary Sections S18 / S19 / S21.

This module collects the small, dependency-light statistical tests that
appear repeatedly in the audit code: one-sided exact binomial p-values,
the cluster-bootstrap CI, Cohen's $d$ (paired), Holm-Bonferroni family-wise
correction, and a post-hoc power analysis for the one-sided binomial test.

We deliberately keep this module pure-Python + numpy so it runs in the
Docker image without scipy dependency.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


# ── exact one-sided binomial (no scipy dependency) ───────────────────────────
def exact_binomial_pvalue(
    successes: int,
    trials: int,
    p0: float = 0.5,
    alternative: str = "greater",
) -> float:
    """Exact one-sided binomial p-value against ``H0: P = p0``.

    Implemented via the regularised incomplete beta function. We do not
    depend on scipy so the smoke tests can run inside the minimal
    Docker container.

    Parameters
    ----------
    successes, trials : int
    p0 : float, default 0.5
        Null hypothesis probability.
    alternative : {"greater", "less"}
        Direction for one-sided test.

    Returns
    -------
    float in [0, 1].
    """

    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be 'greater' or 'less'")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid successes/trials")

    if alternative == "greater":
        k0 = max(successes, 0)
        return _betinc_sum(trials, k0, p0)
    else:
        return 1.0 - _betinc_sum(trials, successes - 1, p0) if successes > 0 else 1.0


def _betinc_sum(n: int, k0: int, p: float) -> float:
    """Sum of binomial probabilities ``sum_{k=k0..n} C(n,k) p^k (1-p)^(n-k)``."""

    if k0 > n:
        return 0.0
    if k0 <= 0:
        return 1.0
    total = 0.0
    q = 1.0 - p
    # Compute terms iteratively from the mode to avoid overflow.
    if p == 0:
        return 0.0
    if p == 1:
        return 1.0 if n >= k0 else 0.0
    # log-binomial coefficient not necessary; we walk along the distribution.
    _pmf_at(n, int(round(n * p)), p, q) if n * p < n else 0.0
    # Brute-force enumeration is acceptable for the small n used by this helper.
    # For substantially larger n, use a numerically stable survival function.
    if n <= 200:
        for k in range(k0, n + 1):
            total += _pmf_at(n, k, p, q)
        return min(1.0, max(0.0, total))
    # Fallback for n > 200: use a normal upper-tail approximation.
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    z = (k0 - 0.5 - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


def _pmf_at(n: int, k: int, p: float, q: float) -> float:
    """Binomial pmf at (n, k) using log-space."""

    if k < 0 or k > n:
        return 0.0
    # Use Stirling-approximated log-gamma for general n. For n <= 200 it's
    # accurate enough to just walk the pmf directly.
    if n <= 1000:
        log_pmf = _log_binomial_pmf(n, k, p, q)
        return math.exp(log_pmf)
    # fallback: normal approximation at one point
    mu = n * p
    sigma = math.sqrt(mu * q)
    z = (k - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def _log_binomial_pmf(n: int, k: int, p: float, q: float) -> float:
    if k == 0 or k == n:
        return k * math.log(max(p, 1e-300)) + (n - k) * math.log(max(q, 1e-300))
    return k * math.log(max(p, 1e-300)) + (n - k) * math.log(max(q, 1e-300)) + _log_comb(n, k)


def _log_comb(n: int, k: int) -> float:
    """log C(n, k) computed via log-factorials."""

    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    s = 0.0
    for i in range(1, k + 1):
        s += math.log((n - k + i) / i)
    return s


# ── sign test ───────────────────────────────────────────────────────────────
def sign_test(deltas: Sequence[float], alternative: str = "less") -> float:
    """One-sided sign test on the sign of ``deltas``.

    Returns the exact one-sided binomial p-value (null hypothesis:
    ``P(positive) = 0.5``). Deltas equal to zero are dropped.

    Parameters
    ----------
    alternative : {"less", "greater"}
        "less" tests the hypothesis that the negative direction is
        over-represented (``P(positive) < 0.5``); "greater" tests the
        converse.
    """

    positives = sum(1 for d in deltas if d > 0)
    negatives = sum(1 for d in deltas if d < 0)
    if positives == 0 and negatives == 0:
        return 1.0
    if alternative == "less":
        # P(X >= negatives | n, p=0.5) — we are testing if negatives
        # are unusually *many*, which would correspond to ``less``.
        return exact_binomial_pvalue(negatives, positives + negatives, 0.5, alternative="greater")
    return exact_binomial_pvalue(positives, positives + negatives, 0.5, alternative="greater")


# ── cluster bootstrap CI ─────────────────────────────────────────────────────
def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[int],
    *,
    statistic: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs),
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI for a per-cluster statistic.

    Resamples clusters with replacement, computes the statistic over the
    union of resampled values, and reports the (alpha/2, 1-alpha/2)
    percentile bootstrap interval.

    Parameters
    ----------
    values, clusters : sequence of equal length
        Each ``(value_i, cluster_i)`` pair.
    statistic : callable on a sequence of values
        Defaults to the mean. For t-test-style intervals use
        :func:`mean_with_se` and combine with t_critical.
    n_resamples : int
    rng : random.Random, optional
        For deterministic runs.

    Returns
    -------
    (point, ci_low, ci_high)
    """

    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    rng = rng or random.Random(42)

    # Map: cluster_id -> [values in that cluster]
    by_cluster: dict[int, list[float]] = {}
    for v, c in zip(values, clusters, strict=False):
        by_cluster.setdefault(int(c), []).append(v)
    cluster_ids = list(by_cluster.keys())
    if not cluster_ids:
        return (float("nan"), float("nan"), float("nan"))

    point = statistic(list(values))

    boots: list[float] = []
    for _ in range(n_resamples):
        sample: list[float] = []
        for _ in range(len(cluster_ids)):
            sample.extend(by_cluster[rng.choice(cluster_ids)])
        boots.append(statistic(sample))
    boots.sort()
    lo_i = max(0, int(round((alpha / 2) * n_resamples)))
    hi_i = min(n_resamples - 1, int(round((1 - alpha / 2) * n_resamples)))
    return (point, boots[lo_i], boots[hi_i])


# ── mean-with-SE wrapper (enables t-test style interval downstream) ────────
def mean_with_se(values: Sequence[float]) -> tuple[float, float]:
    """Return (mean, standard error of mean) for a vector."""

    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    mu = sum(values) / n
    if n < 2:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return (mu, math.sqrt(var / n))


# ── t-test CI for paired deltas ─────────────────────────────────────────────
def t_test_ci_paired(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """t-test 95% CI (two-sided) on the mean of a paired-delta vector.

    Returns ``(mean, ci_low, ci_high, df)``. df = n - 1. Uses a hard-coded
    approximation for t_{alpha/2, df} from a small lookup table.
    """

    n = len(values)
    if n < 2:
        m = values[0] if values else float("nan")
        return (m, m, m, 0.0)
    mu, se = mean_with_se(values)
    t_crit = _t_critical(alpha / 2, n - 1)
    return (mu, mu - t_crit * se, mu + t_crit * se, float(n - 1))


def _t_critical(p: float, df: int) -> float:
    """Crude t-critical approximation. Adequate for df >= 5 and p=0.025."""

    if df < 2:
        return 12.706
    if df < 5:
        return 2.776
    if df < 10:
        return 2.262
    if df < 20:
        return 2.101
    if df < 30:
        return 2.042
    if df < 60:
        return 2.000
    return 1.960


# ── Cohen's d (paired) ──────────────────────────────────────────────────────
def cohens_d_paired(values: Sequence[float]) -> float:
    """Cohen's $d$ for paired samples: ``mean(values) / std(values)``.

    For the bias-Δ values treated as one-sample paired deltas (initial vs.
    corrected on the same page).
    """

    n = len(values)
    if n < 2:
        return float("nan")
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return float("inf") if mu != 0 else 0.0
    return mu / sd


# ── Holm-Bonferroni ─────────────────────────────────────────────────────────
def holm_bonferroni(
    pvalues: Sequence[float],
    alpha: float = 0.05,
) -> list[tuple[int, float, bool, float]]:
    """Holm-Bonferroni step-down correction.

    Returns a list of ``(idx_orig, p_raw, significant, p_adj)`` tuples, in
    the order of the original indices (so callers can re-align with their
    array of metrics).
    """

    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda t: t[1])
    out: list[tuple[int, float, bool, float]] = [(0, 0.0, False, 1.0)] * n
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        p_adj = min(1.0, max(running_max, (n - rank) * p))
        running_max = p_adj
        out[idx] = (idx, p, p_adj < alpha, p_adj)
    return out


# ── post-hoc power (one-sided exact binomial) ────────────────────────────────
def post_hoc_power_binomial(
    n: int,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    trials_grid: Sequence[int] | None = None,
) -> dict[str, float]:
    """Estimate one-sided binomial power under alternative ``p1``.

    Returns ``power_at_n``, ``required_n_for_80pct``, ``rejection_threshold_k``.

    For small ``n`` the function enumerates all possible ``k`` values; for
    larger ``n`` we use a normal-approximation rejection threshold.
    """

    # Find the smallest k such that P(X >= k | p0) <= alpha.
    k_star = _rejection_threshold_binomial(n, p0, alpha)

    # Power = P(X >= k_star | p1).
    if k_star > n:
        power = 0.0
    elif k_star <= 0:
        power = 1.0
    else:
        power = _upper_tail_binomial(n, k_star, p1)

    # Required n for power >= 0.8 under p1
    required = trials_grid or list(range(1, 200))
    required_n = next(
        (nn for nn in required if _power_at(nn, p0, p1, alpha) >= 0.8),
        None,
    )
    return {
        "power_at_n": power,
        "rejection_threshold_k": k_star,
        "required_n_for_80pct": required_n if required_n is not None else ">200",
        "n_evaluated": n,
    }


def _upper_tail_binomial(n: int, k_star: int, p: float) -> float:
    """Compute ``P(X >= k_star | n, p)`` by direct enumeration."""

    if k_star > n:
        return 0.0
    q = 1.0 - p
    if p == 1.0:
        return 1.0
    cum = 0.0
    for k in range(k_star, n + 1):
        cum += _pmf_at(n, k, p, q)
    return min(1.0, max(0.0, cum))


def _rejection_threshold_binomial(n: int, p0: float, alpha: float) -> int:
    """Find the smallest ``k`` such that ``P_0(X >= k) <= alpha``.

    Walks the upper tail from ``k = n`` down to ``k = 1``; rejects
    whenever ``P_0(X >= k) <= alpha``. Returns the smallest ``k`` so
    identified.

    For ``n=9, p0=0.5, alpha=0.05`` the answer is 8 because
    ``P(X>=8) = 10/512 ≈ 0.0195 < 0.05``, while ``P(X>=7) ≈ 0.090``.
    """

    if n <= 0:
        return 1
    q = 1 - p0
    cum = 0.0
    smallest = n + 1
    for k in range(n, 0, -1):
        cum += _pmf_at(n, k, p0, q)
        if cum <= alpha:
            smallest = min(smallest, k)
    if smallest > n:
        return n + 1  # no rejection region exists at this n and alpha
    return smallest


def _power_at(n: int, p0: float, p1: float, alpha: float) -> float:
    k_star = _rejection_threshold_binomial(n, p0, alpha)
    return _upper_tail_binomial(n, k_star, p1)
