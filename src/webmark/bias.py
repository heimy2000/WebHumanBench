"""Composite bias score (Equation 1 of the paper) with both Gaussian and
Wasserstein-1 reference variants.

The composite is a five-dimensional penalty:

    Bias(x) = sum_j  φ( (f_j(x) - mu_j) / sigma_j )

where ``f_j(x)`` is the per-dimension feature statistic and ``mu_j``,
``sigma_j`` are the human-reference stats. The penalty ``φ`` defaults to
``phi(x) = x ** 2`` (quadratic). Pre-specified alternative penalties
(``|x|``, ``min(|x|, 1)``, Huber) are exposed via :func:`compute_composite_bias`.

The default ref is Gaussian on four dimensions + W-1 on typography
(SE-6 partial non-parametric replacement). A ``reference`` keyword can
override the per-dimension map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .features import FEATURE_NAMES, PageFeatures


# ── penalty family ────────────────────────────────────────────────────────────
def phi_quadratic(x: float) -> float:
    return x * x


def phi_l1(x: float) -> float:
    return abs(x)


def phi_clipped(x: float, cap: float = 1.0) -> float:
    return min(abs(x), cap) ** 2  # signed-clipping, monotone in |x|


def phi_huber(x: float, delta: float = 1.5) -> float:
    a = abs(x)
    if a <= delta:
        return 0.5 * a * a
    return delta * (a - 0.5 * delta)


PENALTY_REGISTRY: dict[str, Callable[[float], float]] = {
    "l2": phi_quadratic,
    "l1": phi_l1,
    "huber": phi_huber,
}


# ── reference statistics ───────────────────────────────────────────────────
@dataclass
class ReferenceStats:
    """Per-dimension reference parameters.

    ``means`` and ``stds`` are length-5 (one entry per dimension in
    ``FEATURE_NAMES``). For a W-1 dimension, the corresponding ``stds`` value
    acts as the supplied normalization scale. ``wasserstein_samples_per_dim``
    optionally provides a non-parametric reference for typography or another
    dimension.
    """

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    wasserstein_samples_per_dim: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Mapping[str, float]]) -> ReferenceStats:
        """Convenience constructor accepting a dict-of-dicts (e.g. from JSON)."""
        means = {k: float(v["mean"]) for k, v in d.items()}
        stds = {k: float(v["std"]) for k, v in d.items()}
        return cls(means=means, stds=stds)


# ── z-score and W-1 primitives ──────────────────────────────────────────────
def gaussian_zscore(value: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return (value - mu) / sigma


def wasserstein1_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Empirical 1-Wasserstein distance between two sorted samples.

    W1(P, Q) = integral over |F_P^{-1}(u) - F_Q^{-1}(u)| du,
    approximated by the discrete mean of absolute quantile differences.
    For two sorted samples of equal length, the unweighted mean equals
    the L1 mean of the per-quantile absolute differences divided by N.
    """
    if not p or not q:
        return 0.0
    p_sorted = sorted(p)
    q_sorted = sorted(q)
    if len(p_sorted) == len(q_sorted):
        diffs = (abs(a - b) for a, b in zip(p_sorted, q_sorted, strict=False))
        return sum(diffs) / len(p_sorted)
    # unequal lengths: interpolate q to length of p
    import numpy as np
    p_arr = np.asarray(p_sorted, dtype=float)
    q_arr = np.asarray(q_sorted, dtype=float)
    p_q = np.linspace(0.0, 1.0, len(p_arr))
    q_q = np.linspace(0.0, 1.0, len(q_arr))
    q_interp = np.interp(p_q, q_q, q_arr)
    return float(np.mean(np.abs(p_arr - q_interp)))


# ── composite scoring ──────────────────────────────────────────────────────
@dataclass
class BiasComponents:
    """Decomposed bias score: per-dimension and total."""

    per_dim: dict[str, float]
    total: float

    def to_dict(self) -> dict[str, float]:
        return {"per_dim": self.per_dim, "total": self.total}


class BiasScorer:
    """Stateless composite-bias scorer; thread-safe."""

    def __init__(
        self,
        reference: ReferenceStats,
        penalty: str = "l2",
        nonparametric_dims: Sequence[str] | None = None,
    ) -> None:
        self.reference = reference
        self.penalty = penalty
        self.nonparametric_dims = set(nonparametric_dims or ())

    def score(self, features: PageFeatures) -> BiasComponents:
        """Compute the bias vector for a single page."""

        per_dim: dict[str, float] = {}
        for dim in FEATURE_NAMES:
            mu = self.reference.means.get(dim, 0.0)
            sigma = self.reference.stds.get(dim, 1.0)

            # A zero-variance train-only statistic cannot support a normalized
            # distance without inventing an arbitrary scale. Treat it as
            # unidentifiable rather than turning a small pilot into a brittle
            # exact-match rule.
            if sigma <= 0:
                per_dim[dim] = 0.0
                continue

            # Decide between Gaussian (default) and W-1 (typography, SE-6).
            if dim in self.nonparametric_dims or dim in self.reference.wasserstein_samples_per_dim:
                ref_samples = self.reference.wasserstein_samples_per_dim.get(dim, [mu])
                # Aggregate the page-side distribution to a comparable summary.
                page_samples = self._page_distribution(dim, features)
                z = wasserstein1_distance(page_samples, ref_samples) / sigma
            else:
                v = self._page_summary(dim, features)
                z = gaussian_zscore(v, mu, sigma)
            per_dim[dim] = _apply_penalty(self.penalty, z)

        total = sum(per_dim.values())
        return BiasComponents(per_dim=per_dim, total=total)

    def delta(self, initial: PageFeatures, corrected: PageFeatures) -> BiasComponents:
        """Compute Δ = bias(corrected) - bias(initial). Negative is improvement."""

        init = self.score(initial)
        corr = self.score(corrected)
        per_dim = {k: corr.per_dim[k] - init.per_dim[k] for k in FEATURE_NAMES}
        return BiasComponents(per_dim=per_dim, total=corr.total - init.total)

    @staticmethod
    def _page_summary(dim: str, features: PageFeatures) -> float:
        arr = getattr(features, dim, [])
        if not arr:
            return 0.0
        # ``color`` is a string palette list; reduce to its unique-count.
        if dim == "color":
            return float(len({c.lower() for c in arr if c}))
        return sum(arr) / len(arr)

    @staticmethod
    def _page_distribution(dim: str, features: PageFeatures) -> list[float]:
        """Return the empirical distribution for a dimension.

        For typography we return the raw samples (W-1 is meaningful here).
        Other dimensions currently only have a per-page summary; we
        replicate it five times to make W-1 well-defined. ``color`` is
        reduced to a count which is then replicated.
        """
        arr = getattr(features, dim, [])
        if dim == "typography":
            return list(arr)
        if not arr:
            return []
        if dim == "color":
            mu = float(len({c.lower() for c in arr if c}))
        else:
            mu = sum(arr) / len(arr)
        return [mu] * 5


def _apply_penalty(name: str, z: float) -> float:
    fn = PENALTY_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"unknown penalty {name!r}; choose from {list(PENALTY_REGISTRY)}")
    return fn(z)


def compute_composite_bias(
    features: PageFeatures,
    reference: ReferenceStats,
    penalty: str = "l2",
    nonparametric_dims: Sequence[str] | None = ("typography",),
) -> float:
    """Convenience wrapper for a single-page composite bias score."""

    return BiasScorer(
        reference=reference,
        penalty=penalty,
        nonparametric_dims=nonparametric_dims,
    ).score(features).total
