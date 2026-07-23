"""Bounded beam-search correction.

Default configuration matches the bounded-search setting in the paper:
``K=3`` candidates per step, ``L=5`` nominal maximum depth, no repeated
scored operator, and a static contrast-exposure guardrail. With three scored
release operators, the effective maximum chain length is three.

Determinism: the beam is rank-ordered by the bias-reduction objective
``Δ = bias(corrected) - bias(initial)`` (more negative = better). All
random sampling in this file goes through the injected ``rng`` to keep
runs reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .bias import BiasScorer
from .contrast import static_low_contrast_exposure
from .features import PageFeatures, extract_page_features
from .operators import (
    PRIMARY_OPERATORS,
    OperatorName,
    apply_operator,
)


@dataclass
class BeamSearchConfig:
    """Beam-search configuration parameters."""

    beam_width: int = 3           # K
    max_depth: int = 5            # L
    operators: Sequence[OperatorName] = field(
        default_factory=lambda: tuple(PRIMARY_OPERATORS)
    )
    seed: int = 42
    allow_operator_reuse: bool = False
    enforce_contrast_guardrail: bool = True
    contrast_exposure_tolerance: int = 0


@dataclass
class BeamNode:
    """One node in the beam: an operator chain and its current score."""

    chain: list[OperatorName] = field(default_factory=list)
    score: float = 0.0


def beam_search_correct(
    html: str,
    bias_scorer: BiasScorer,
    *,
    config: BeamSearchConfig | None = None,
    initial_features: PageFeatures | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> tuple[list[OperatorName], float]:
    """Run beam search to find an operator chain reducing the bias score.

    Parameters
    ----------
    html : str
        The starting (initial) HTML page.
    bias_scorer : BiasScorer
        Composite-bias scorer; passed in so the same scorer instance can be
        reused across the whole sweep.
    config : BeamSearchConfig, optional
        Defaults to the paper's K=3, nominal L=5 bounded configuration.
    initial_features : PageFeatures, optional
        Pre-computed features for the initial page (avoids re-extraction).
    progress_hook : callable, optional
        ``progress_hook(step, total_steps)`` called after every depth step;
        useful for live progress bars.

    Returns
    -------
    (chain, delta_score) where ``chain`` is the optimal operator sequence
    and ``delta_score = bias(corrected) - bias(initial)`` (negative is better).
    """

    cfg = config or BeamSearchConfig()
    initial_html = html
    init_feats = initial_features or extract_page_features(initial_html)
    init_score = bias_scorer.score(init_feats).total
    init_contrast_exposure = (
        static_low_contrast_exposure(initial_html)
        if cfg.enforce_contrast_guardrail
        else 0
    )

    best_node = BeamNode(chain=[], score=0.0)
    frontier: list[BeamNode] = [best_node]

    for step in range(1, cfg.max_depth + 1):
        candidates: list[BeamNode] = []
        for node in frontier:
            for op in cfg.operators:
                if not cfg.allow_operator_reuse and op in node.chain:
                    continue
                new_chain = node.chain + [op]
                new_html = _apply_chain_to_html(initial_html, new_chain)
                if cfg.enforce_contrast_guardrail:
                    exposure = static_low_contrast_exposure(new_html)
                    if exposure > init_contrast_exposure + cfg.contrast_exposure_tolerance:
                        continue
                feats = extract_page_features(new_html)
                score_after = bias_scorer.score(feats).total
                delta = score_after - init_score
                cand = BeamNode(chain=new_chain, score=delta)
                candidates.append(cand)

        if not candidates:
            break

        # Rank by score (more negative = better) and keep top-K.
        candidates.sort(key=lambda n: n.score)
        frontier = candidates[: max(1, cfg.beam_width)]

        if frontier and frontier[0].score < best_node.score:
            best_node = frontier[0]

        if progress_hook is not None:
            progress_hook(step, cfg.max_depth)

    return best_node.chain, best_node.score


# ── helpers ──────────────────────────────────────────────────────────────────
def _apply_chain_to_html(initial_html: str, chain: Sequence[OperatorName]) -> str:
    out = initial_html
    for op in chain:
        out = apply_operator(out, op)
    return out
