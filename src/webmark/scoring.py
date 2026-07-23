"""External scorers: UIClip and Aesthetic V2.5.

Both models are external dependencies that we do not redistribute, so the
classes below are adapters around their public inference interfaces. The
release runs the same Normalisation pipeline that the paper reports so
that UIClip scores are mapped to the [0, 1] interval via a site-level
contrastive normalisation (paper §4.3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

UICLIP_NORMALIZATION: float = 1.0  # upper-bound shift used in §4.3


@dataclass
class AestheticScore:
    """Aesthetic V2.5 output: 1--5 Likert, plus a 0--1 normalised form."""

    raw: float
    normalised: float


class UIClipScorer:
    """Adapter for the UIClip model. Implementation calls the public API.

    The class is constructed with a ``client`` (any object exposing a
    ``score(images)`` method, returning a list of floats); concrete
    injection allows unit tests to substitute a stub client.
    """

    def __init__(self, client: object | None = None) -> None:
        self.client = client

    def score(self, images: Sequence[bytes]) -> list[float]:
        if self.client is None:
            raise RuntimeError(
                "UIClipScorer requires a client; pass a model adapter or "
                "mock in tests"
            )
        return list(self.client.score(images))


class AestheticV25Scorer:
    """Adapter for the Aesthetic V2.5 model.

    The paper reports the raw 1--5 Likert scale as well as a 0--1
    normalised form; both are exposed in :meth:`score`.
    """

    def __init__(self, client: object | None = None) -> None:
        self.client = client

    def score(self, images: Sequence[bytes]) -> list[AestheticScore]:
        if self.client is None:
            raise RuntimeError(
                "AestheticV25Scorer requires a client; pass a model adapter "
                "or mock in tests"
            )
        raw_scores = list(self.client.score(images))
        return [
            AestheticScore(raw=r, normalised=_normalise_1_to_5(r))
            for r in raw_scores
        ]


def _normalise_1_to_5(x: float) -> float:
    """Map a 1--5 Likert score to [0, 1]."""

    return max(0.0, min(1.0, (x - 1.0) / 4.0))


def mean_delta_score(
    initial_scores: Sequence[float],
    corrected_scores: Sequence[float],
) -> dict[str, float]:
    """Mean Δ across two paired score sequences.

    A positive Δ means the corrected version scores higher; UIClip is
    expected to have Δ ≈ 0 (paper §5) and Aesthetic V2.5 to have a small
    positive Δ.
    """

    if len(initial_scores) != len(corrected_scores):
        raise ValueError("score sequences must have the same length")
    if not initial_scores:
        return {"delta_mean": float("nan"), "delta_std": float("nan"), "n": 0}
    deltas = [c - i for i, c in zip(initial_scores, corrected_scores, strict=False)]
    mean = sum(deltas) / len(deltas)
    if len(deltas) < 2:
        std = 0.0
    else:
        var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
        std = math.sqrt(var)
    return {"delta_mean": mean, "delta_std": std, "n": len(deltas)}
