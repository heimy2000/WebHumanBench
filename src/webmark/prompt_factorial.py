"""Matched prompt-factorial diagnostics for WebHumanBench.

The experiment crosses two disclosed generation instructions: requesting
literal CSS values readable by the extractor and requesting a plausible
human-authored style.  It is a prompt-sensitivity diagnostic, not an
authorship or preference experiment.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from .benchmark_baselines import _stratified_bootstrap_auc

PROMPT_FACTORIAL_SCHEMA = "webmark_prompt_factorial_v1"
PAGE_TYPES = (
    "saas_landing",
    "docs_homepage",
    "product_showcase",
    "developer_tool",
    "dashboard_shell",
    "portfolio_showcase",
)
CONDITIONS = {
    "neutral": {"literal_css": False, "human_style": False},
    "literal_only": {"literal_css": True, "human_style": False},
    "style_only": {"literal_css": False, "human_style": True},
    "full": {"literal_css": True, "human_style": True},
}
PAGE_TYPE_BRIEFS = {
    "saas_landing": (
        "a SaaS landing page with a hero, feature cards, pricing teaser, testimonials, "
        "and a final call-to-action"
    ),
    "docs_homepage": (
        "a documentation homepage with a product intro, quick-start steps, doc cards, "
        "API reference links, and a search affordance"
    ),
    "product_showcase": (
        "a product showcase page with narrative hero copy, product highlights, gallery "
        "cards, specifications, and a purchase call-to-action"
    ),
    "developer_tool": (
        "a developer-tool homepage with terminal or code snippets, integration cards, "
        "workflow steps, and documentation actions"
    ),
    "dashboard_shell": (
        "a dashboard shell with sidebar navigation, metric cards, table or list content, "
        "filters, and compact controls"
    ),
    "portfolio_showcase": (
        "a creative portfolio page with project tiles, a biography section, case-study "
        "links, and a contact call-to-action"
    ),
}
SCENARIOS = (
    "privacy-first analytics",
    "AI coding assistant",
    "open-source design system",
    "climate-data platform",
    "collaborative notes workspace",
    "robotics monitoring console",
    "medical scheduling tool",
    "fintech reconciliation app",
    "education course builder",
    "creator-commerce storefront",
    "security observability product",
    "research lab homepage",
)


@dataclass(frozen=True)
class FactorialPage:
    page_id: str
    block_id: str
    page_type: str
    condition: str
    literal_css: bool
    human_style: bool
    model: str
    index: int
    seed: int
    scenario: str


def build_factorial_plan(
    *,
    model: str,
    blocks_per_type: int,
    seed: int,
) -> list[FactorialPage]:
    """Return a balanced, deterministic plan before any generation calls."""
    if not model.strip():
        raise ValueError("model must be non-empty")
    if blocks_per_type <= 0:
        raise ValueError("blocks_per_type must be positive")
    pages: list[FactorialPage] = []
    for type_index, page_type in enumerate(PAGE_TYPES):
        for block_index in range(blocks_per_type):
            block_seed = seed + type_index * 10_000 + block_index
            scenario = SCENARIOS[random.Random(block_seed).randrange(len(SCENARIOS))]
            block_id = f"{page_type}:{block_index:02d}"
            for condition, factors in CONDITIONS.items():
                pages.append(
                    FactorialPage(
                        page_id=f"pf_{page_type}_{block_index:02d}_{condition}",
                        block_id=block_id,
                        page_type=page_type,
                        condition=condition,
                        literal_css=factors["literal_css"],
                        human_style=factors["human_style"],
                        model=model,
                        index=block_index,
                        seed=block_seed,
                        scenario=scenario,
                    )
                )
    return pages


def prompt_messages(page: FactorialPage) -> list[dict[str, str]]:
    """Build a prompt whose only experimental changes are the two factors."""
    if page.condition not in CONDITIONS:
        raise ValueError(f"unknown prompt condition {page.condition!r}")
    expected = CONDITIONS[page.condition]
    if (page.literal_css, page.human_style) != (
        expected["literal_css"],
        expected["human_style"],
    ):
        raise ValueError("page factors do not match its named condition")

    constraints = [
        f"- Scenario/theme: {page.scenario}",
        f"- Page type label: {page.page_type}",
        "- Output one complete HTML document with inline CSS.",
        "- Do not use external assets, scripts, icon libraries, web fonts, iframes, or network calls.",
        "- Use semantic landmarks where natural and include at least one link or button.",
        "- Use text or inline CSS shapes for image-like content and provide accessible labels.",
        "- Keep the HTML/CSS source concise enough to render without a build step.",
    ]
    if page.literal_css:
        constraints.append(
            "- Use literal CSS values for font sizes, line heights, spacing, radii, and colors; "
            "avoid CSS variables for primary visual tokens."
        )
    if page.human_style:
        constraints.append(
            "- Make the design plausible for a modern human-authored website without copying a named brand."
        )

    system = (
        "Generate a self-contained webpage for a controlled research experiment. Return only one "
        "complete HTML document, without Markdown fences or explanation."
    )
    user = "\n".join(
        [
            f"Create {PAGE_TYPE_BRIEFS[page.page_type]}.",
            "",
            "Fixed constraints:",
            *constraints,
            "",
            f"Variation seed: {page.seed}",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def plan_as_dicts(plan: list[FactorialPage]) -> list[dict[str, Any]]:
    return [asdict(page) for page in plan]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _win_fraction(values: list[float]) -> float:
    return statistics.fmean(1.0 if value < 0 else 0.5 if value == 0 else 0.0 for value in values)


def _paired_effects(
    score_rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    distances: dict[tuple[str, str, str], float] = {}
    for row in score_rows:
        group_id = str(row["group_id"])
        if group_id not in metadata:
            continue
        item = metadata[group_id]
        key = (str(item["page_type"]), str(item["block_id"]), str(item["condition"]))
        if key in distances:
            raise ValueError(f"duplicate score for {key!r}")
        distances[key] = float(row["distance"])

    complete: dict[str, dict[str, dict[str, float]]] = {}
    for page_type in PAGE_TYPES:
        blocks: dict[str, dict[str, float]] = {}
        block_ids = sorted({block for observed_type, block, _ in distances if observed_type == page_type})
        for block_id in block_ids:
            condition_scores = {
                condition: distances[(page_type, block_id, condition)]
                for condition in CONDITIONS
                if (page_type, block_id, condition) in distances
            }
            if set(condition_scores) == set(CONDITIONS):
                blocks[block_id] = condition_scores
        complete[page_type] = blocks

    if any(not blocks for blocks in complete.values()):
        missing = [page_type for page_type, blocks in complete.items() if not blocks]
        raise ValueError(f"factorial analysis lacks a complete block for: {', '.join(missing)}")

    deltas: dict[str, dict[str, list[float]]] = {
        "literal_css": {},
        "human_style": {},
        "interaction": {},
    }
    for page_type, blocks in complete.items():
        literal_values: list[float] = []
        style_values: list[float] = []
        interaction_values: list[float] = []
        for scores in blocks.values():
            literal_values.append(
                (scores["literal_only"] + scores["full"]) / 2
                - (scores["neutral"] + scores["style_only"]) / 2
            )
            style_values.append(
                (scores["style_only"] + scores["full"]) / 2
                - (scores["neutral"] + scores["literal_only"]) / 2
            )
            interaction_values.append(
                scores["full"]
                - scores["literal_only"]
                - scores["style_only"]
                + scores["neutral"]
            )
        deltas["literal_css"][page_type] = literal_values
        deltas["human_style"][page_type] = style_values
        deltas["interaction"][page_type] = interaction_values

    rng = random.Random(seed)
    effects: dict[str, Any] = {}
    for factor in ("literal_css", "human_style"):
        by_type = deltas[factor]
        point = statistics.fmean(_win_fraction(values) for values in by_type.values())
        samples: list[float] = []
        for _ in range(n_resamples):
            type_wins = []
            for values in by_type.values():
                sample = [rng.choice(values) for _ in values]
                type_wins.append(_win_fraction(sample))
            samples.append(statistics.fmean(type_wins))
        effects[factor] = {
            "equal_page_type_paired_closer_fit_rate": point,
            "ci_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
            "n_complete_blocks": sum(len(values) for values in by_type.values()),
            "by_page_type": {
                page_type: {
                    "n_blocks": len(values),
                    "closer_fit_rate": _win_fraction(values),
                    "median_distance_delta": statistics.median(values),
                }
                for page_type, values in by_type.items()
            },
        }
    effects["interaction"] = {
        "definition": "D_full-D_literal_only-D_style_only+D_neutral",
        "by_page_type": {
            page_type: {
                "n_blocks": len(values),
                "median_distance_delta": statistics.median(values),
            }
            for page_type, values in deltas["interaction"].items()
        },
    }
    return {
        "distance_direction": "lower_is_closer_to_the_declared_reference",
        "factor_delta_definition": "mean_distance_factor_on-minus-factor_off_within_matched_block",
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "unit": "matched_block_stratified_by_page_type",
        },
        "complete_blocks_by_page_type": {
            page_type: len(blocks) for page_type, blocks in complete.items()
        },
        "effects": effects,
    }


def _feature_diagnostics(
    records: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows_by_condition: dict[str, list[dict[str, float]]] = {condition: [] for condition in CONDITIONS}
    for record in records:
        group_id = str(record["group_id"])
        if group_id not in metadata:
            continue
        condition = str(metadata[group_id]["condition"])
        features = record["features"]
        spacing = [float(value) for value in features["spacing"]]
        colors = [str(value).lower() for value in features["color"] if value]
        rows_by_condition[condition].append(
            {
                "spacing_fraction_1_2": statistics.fmean(
                    1.0 if abs(value - 1.2) <= 1e-9 else 0.0 for value in spacing
                ),
                "spacing_mean": statistics.fmean(spacing),
                "unique_colors": float(len(set(colors))),
                "color_samples": float(len(colors)),
            }
        )
    return {
        condition: {
            "n_groups": len(rows),
            "median_group_spacing_fraction_1_2": statistics.median(
                row["spacing_fraction_1_2"] for row in rows
            ),
            "median_spacing_mean": statistics.median(row["spacing_mean"] for row in rows),
            "median_unique_colors": statistics.median(row["unique_colors"] for row in rows),
            "median_color_samples": statistics.median(row["color_samples"] for row in rows),
        }
        for condition, rows in rows_by_condition.items()
        if rows
    }


def analyze_prompt_factorial(
    baseline_result: dict[str, Any],
    records: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    n_resamples: int = 2_000,
    seed: int = 91,
) -> dict[str, Any]:
    """Analyze condition endpoints and matched effects for all supplied baselines."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    metadata = {
        f"generation-{row['page_id']}": row
        for row in plan
    }
    analyses: dict[str, Any] = {}
    for baseline_index, (baseline, payload) in enumerate(
        sorted(baseline_result["baselines"].items())
    ):
        score_rows = payload["groups"]
        historical = [row for row in score_rows if row["source"] == "human"]
        condition_endpoints: dict[str, Any] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            condition_rows = [
                row
                for row in score_rows
                if row["group_id"] in metadata
                and metadata[row["group_id"]]["condition"] == condition
            ]
            condition_endpoints[condition] = _stratified_bootstrap_auc(
                historical + condition_rows,
                n_resamples=n_resamples,
                seed=seed + baseline_index * 100 + condition_index,
                min_groups_for_ci=10,
            )
        analyses[baseline] = {
            "condition_type_macro_auroc": condition_endpoints,
            "paired_factor_effects": _paired_effects(
                score_rows,
                metadata,
                n_resamples=n_resamples,
                seed=seed + baseline_index,
            ),
        }
    return {
        "schema": PROMPT_FACTORIAL_SCHEMA,
        "design": {
            "factors": CONDITIONS,
            "page_types": list(PAGE_TYPES),
            "matching_unit": "page_type_scenario_variation_seed_block",
        },
        "baselines": analyses,
        "feature_diagnostics": _feature_diagnostics(records, metadata),
        "claim_boundary": (
            "This experiment isolates two disclosed prompt clauses for one recorded model endpoint. "
            "It does not identify a model-general style effect, validate human authorship, or remove "
            "temporal and source-selection confounding in the historical-reference cohort."
        ),
    }
