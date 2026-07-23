"""Sensitivity and exclusion diagnostics for the WebHumanBench v1 pilot.

The diagnostics in this module operate only on frozen release artifacts. They
do not add perceptual labels or reinterpret operational provenance as verified
authorship.
"""
from __future__ import annotations

import copy
import hashlib
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .benchmark_baselines import (
    BASELINE_NAMES,
    _group_rows,
    _percentile,
    _profile_scores,
    _reference_parameters,
    _score_simple_baseline,
    _stratified_bootstrap_auc,
    evaluate_reference_fit_baselines,
)
from .features import FEATURE_NAMES
from .human_likeness import validate_manifest

DIAGNOSTICS_SCHEMA = "webmark_webhumanbench_v1_diagnostics_v3"


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("numeric summary requires at least one value")
    numeric = [float(value) for value in values]
    return {
        "median": statistics.median(numeric),
        "q1": _percentile(numeric, 0.25),
        "q3": _percentile(numeric, 0.75),
        "min": min(numeric),
        "max": max(numeric),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, including deterministic tie handling."""
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equally sized sequences of length >= 2")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if len(set(left_ranks)) == 1 or len(set(right_ranks)) == 1:
        return None
    return statistics.correlation(left_ranks, right_ranks)


def _human_groups_by_type(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in manifest.get("records", []):
        if record.get("source") == "human":
            grouped[str(record["page_type"])].add(str(record["group_id"]))
    if not grouped:
        raise ValueError("manifest has no historical-reference groups")
    return {page_type: sorted(group_ids) for page_type, group_ids in grouped.items()}


def _split_counts_by_type(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    for record in manifest.get("records", []):
        if record.get("source") != "human":
            continue
        group_id = str(record["group_id"])
        if group_id in seen:
            continue
        seen.add(group_id)
        counts[str(record["page_type"])][str(record["split"])] += 1
    output = {
        page_type: {split: counter.get(split, 0) for split in ("train", "dev", "test")}
        for page_type, counter in counts.items()
    }
    for page_type, page_counts in output.items():
        if page_counts["train"] < 2 or page_counts["test"] < 1:
            raise ValueError(
                f"page type {page_type!r} requires at least two train and one test group"
            )
    return output


def _assignment_signature(assignments: Mapping[str, str]) -> str:
    payload = "\n".join(f"{group_id}\t{assignments[group_id]}" for group_id in sorted(assignments))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frozen_assignments(manifest: Mapping[str, Any]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for record in manifest.get("records", []):
        if record.get("source") != "human":
            continue
        group_id = str(record["group_id"])
        split = str(record["split"])
        prior = assignments.setdefault(group_id, split)
        if prior != split:
            raise ValueError(f"historical group {group_id!r} crosses splits")
    return assignments


def _alternative_assignment(
    groups_by_type: Mapping[str, Sequence[str]],
    counts_by_type: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
    attempt: int,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for page_type in sorted(groups_by_type):
        group_ids = list(groups_by_type[page_type])
        rng = random.Random(_stable_seed(seed, attempt, page_type))
        rng.shuffle(group_ids)
        n_train = int(counts_by_type[page_type]["train"])
        n_dev = int(counts_by_type[page_type]["dev"])
        for index, group_id in enumerate(group_ids):
            if index < n_train:
                split = "train"
            elif index < n_train + n_dev:
                split = "dev"
            else:
                split = "test"
            assignments[group_id] = split
    return assignments


def _manifest_with_assignments(
    manifest: Mapping[str, Any], assignments: Mapping[str, str]
) -> dict[str, Any]:
    records: list[Mapping[str, Any]] = []
    for record in manifest["records"]:
        if record.get("source") == "human":
            updated = dict(record)
            updated["split"] = assignments[str(record["group_id"])]
            records.append(updated)
        else:
            records.append(record)
    return {
        "schema": manifest["schema"],
        "metadata": copy.deepcopy(manifest["metadata"]),
        "records": records,
    }


def evaluate_split_sensitivity(
    manifest: Mapping[str, Any],
    *,
    n_alternative_splits: int = 100,
    seed: int = 2027,
) -> dict[str, Any]:
    """Refit all baselines over deterministic stratified source splits."""
    if n_alternative_splits <= 0:
        raise ValueError("n_alternative_splits must be positive")
    groups_by_type = _human_groups_by_type(manifest)
    counts_by_type = _split_counts_by_type(manifest)
    frozen_assignments = _frozen_assignments(manifest)
    frozen_signature = _assignment_signature(frozen_assignments)

    frozen_result = evaluate_reference_fit_baselines(
        manifest,
        n_resamples=1,
        seed=seed,
        min_groups_for_ci=10_000,
    )
    frozen_aurocs = {
        name: float(frozen_result["baselines"][name]["type_macro"]["point_estimate"])
        for name in BASELINE_NAMES
    }

    runs: list[dict[str, Any]] = []
    signatures = {frozen_signature}
    max_attempts = max(10_000, n_alternative_splits * 200)
    for attempt in range(max_attempts):
        assignments = _alternative_assignment(
            groups_by_type,
            counts_by_type,
            seed=seed,
            attempt=attempt,
        )
        signature = _assignment_signature(assignments)
        if signature in signatures:
            continue
        signatures.add(signature)
        reassigned = _manifest_with_assignments(manifest, assignments)
        result = evaluate_reference_fit_baselines(
            reassigned,
            n_resamples=1,
            seed=seed,
            min_groups_for_ci=10_000,
        )
        runs.append({
            "index": len(runs) + 1,
            "assignment_sha256": signature,
            "aurocs": {
                name: float(result["baselines"][name]["type_macro"]["point_estimate"])
                for name in BASELINE_NAMES
            },
            "inactive_zero_variance_dimensions": {
                page_type: diagnostics["inactive_zero_variance_dimensions"]
                for page_type, diagnostics in result["reference_scale_diagnostics"].items()
            },
        })
        if len(runs) == n_alternative_splits:
            break
    if len(runs) != n_alternative_splits:
        raise ValueError(
            f"could construct only {len(runs)} unique alternative splits after {max_attempts} attempts"
        )

    baseline_summary: dict[str, Any] = {}
    for name in BASELINE_NAMES:
        values = [float(run["aurocs"][name]) for run in runs]
        frozen = frozen_aurocs[name]
        baseline_summary[name] = {
            **_numeric_summary(values),
            "frozen_split": frozen,
            "frozen_percentile_among_alternatives": (
                100.0 * (sum(value < frozen for value in values) + 0.5 * sum(value == frozen for value in values))
                / len(values)
            ),
            "fraction_below_0_5": sum(value < 0.5 for value in values) / len(values),
        }

    zero_counts: dict[str, Counter[str]] = {
        page_type: Counter() for page_type in groups_by_type
    }
    any_zero_counts: Counter[str] = Counter()
    for run in runs:
        for page_type, dimensions in run["inactive_zero_variance_dimensions"].items():
            if dimensions:
                any_zero_counts[page_type] += 1
            zero_counts[page_type].update(dimensions)
    zero_variance_frequency = {
        page_type: {
            "any_dimension": {
                "count": any_zero_counts[page_type],
                "fraction": any_zero_counts[page_type] / len(runs),
            },
            "by_dimension": {
                dimension: {
                    "count": zero_counts[page_type][dimension],
                    "fraction": zero_counts[page_type][dimension] / len(runs),
                }
                for dimension in FEATURE_NAMES
            },
        }
        for page_type in sorted(groups_by_type)
    }
    return {
        "design": "deterministic_stratified_source_group_resplitting",
        "endpoint": "equal_page_type_macro_auroc",
        "seed": seed,
        "n_alternative_splits": len(runs),
        "frozen_assignment_sha256": frozen_signature,
        "counts_by_page_type": counts_by_type,
        "baseline_auroc_summary": baseline_summary,
        "zero_variance_frequency": zero_variance_frequency,
        "runs": runs,
        "interpretation": (
            "Variation across these source-level refits measures split sensitivity within the frozen pilot. "
            "It is not uncertainty over the population of websites and does not remove temporal or prompt confounding."
        ),
    }


def _sourcewise_coverage(
    heldout_distance: float, challenge_distances: Sequence[float]
) -> float:
    """Return the fraction of challenge groups farther away than one held-out source."""
    if not challenge_distances:
        raise ValueError("sourcewise coverage requires challenge distances")
    farther = sum(distance > heldout_distance for distance in challenge_distances)
    tied = sum(distance == heldout_distance for distance in challenge_distances)
    return (farther + 0.5 * tied) / len(challenge_distances)


def _summarize_crossfit_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cross-fit coverage summary requires source rows")
    by_page_type = {
        page_type: [
            float(row["coverage_fraction"])
            for row in rows
            if str(row["page_type"]) == page_type
        ]
        for page_type in sorted({str(row["page_type"]) for row in rows})
    }
    values = [float(row["coverage_fraction"]) for row in rows]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        sampled = [
            rng.choice(type_values)
            for type_values in by_page_type.values()
            for _ in type_values
        ]
        samples.append(statistics.fmean(sampled))
    return {
        "mean": statistics.fmean(values),
        **_numeric_summary(values),
        "ci_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "fraction_below_0_5": sum(value < 0.5 for value in values) / len(values),
        "n_reference_sources": len(values),
        "n_resamples": n_resamples,
        "resampling_unit": "historical_reference_source_stratified_by_page_type",
        "by_page_type": {
            page_type: {
                "n_reference_sources": len(type_values),
                "mean": statistics.fmean(type_values),
                **_numeric_summary(type_values),
            }
            for page_type, type_values in by_page_type.items()
        },
    }


def evaluate_leave_one_reference_out(
    manifest: Mapping[str, Any],
    *,
    n_resamples: int = 2_000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Cross-fit each historical source against same-type generated groups.

    Each held-out source is evaluated under a profile fitted without that source.
    The sourcewise endpoint is its midrank coverage against the fixed generated
    challenge groups of the same page type. Profiles differ across held-out
    sources, so the aggregate is explicitly a mean of sourcewise coverages, not
    one pooled AUROC.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    records = validate_manifest(manifest)
    historical = [record for record in records if record.source == "human"]
    challenge = [record for record in records if record.source == "ai"]
    historical_groups = _group_rows(historical)
    historical_by_type: dict[str, list[str]] = defaultdict(list)
    for row in historical_groups:
        historical_by_type[str(row["page_type"])].append(str(row["group_id"]))

    source_rows: dict[str, list[dict[str, Any]]] = {
        baseline: [] for baseline in BASELINE_NAMES
    }
    for page_type in sorted(historical_by_type):
        reference_ids = sorted(historical_by_type[page_type])
        if len(reference_ids) < 3:
            raise ValueError(
                f"leave-one-reference-out requires at least three sources for {page_type!r}"
            )
        challenge_records = [record for record in challenge if record.page_type == page_type]
        if not challenge_records:
            raise ValueError(f"leave-one-reference-out has no challenge rows for {page_type!r}")
        for heldout_id in reference_ids:
            train_records = [
                replace(record, split="train")
                for record in historical
                if record.page_type == page_type and record.group_id != heldout_id
            ]
            test_records = [
                replace(record, split="test")
                for record in historical
                if record.group_id == heldout_id
            ] + challenge_records
            train_groups = _group_rows(train_records)
            test_groups = _group_rows(test_records)
            parameters = _reference_parameters(train_groups)
            profile_scores = _profile_scores(train_records, test_records)

            for baseline in BASELINE_NAMES:
                distances: dict[str, float] = {}
                for group in test_groups:
                    group_id = str(group["group_id"])
                    if baseline == "profile_l2_w1":
                        distance = float(profile_scores[group_id])
                    else:
                        distance = float(_score_simple_baseline(baseline, group, parameters))
                    distances[group_id] = distance
                heldout_distance = distances[heldout_id]
                challenge_distances = [
                    distance
                    for group_id, distance in distances.items()
                    if group_id != heldout_id
                ]
                source_rows[baseline].append({
                    "group_id": heldout_id,
                    "page_type": page_type,
                    "n_training_reference_sources": len(reference_ids) - 1,
                    "n_generated_challenge_groups": len(challenge_distances),
                    "heldout_distance": heldout_distance,
                    "coverage_fraction": _sourcewise_coverage(
                        heldout_distance, challenge_distances
                    ),
                })

    summaries = {
        baseline: _summarize_crossfit_coverage(
            rows,
            n_resamples=n_resamples,
            seed=_stable_seed(seed, "leave_one_reference_out", baseline),
        )
        for baseline, rows in source_rows.items()
    }
    return {
        "design": "leave_one_historical_reference_source_out_within_page_type",
        "endpoint": (
            "For each held-out historical source, the midrank fraction of same-type generated "
            "challenge groups with a larger distance from a profile fitted on all remaining "
            "historical sources of that type."
        ),
        "n_historical_reference_sources": len(historical_groups),
        "n_bootstrap_resamples": n_resamples,
        "baselines": {
            baseline: {
                "summary": summaries[baseline],
                "sources": source_rows[baseline],
            }
            for baseline in BASELINE_NAMES
        },
        "interpretation": (
            "This cross-fit diagnostic uses all historical sources as held-out units and removes "
            "dependence on one frozen test assignment. It still compares the same temporally and "
            "prompt-confounded cohorts, and its sourcewise profiles are not one pooled classifier."
        ),
    }


def evaluate_model_strata(
    baseline_result: Mapping[str, Any],
    *,
    n_resamples: int = 2_000,
    seed: int = 2027,
    min_groups_for_ci: int = 10,
) -> dict[str, Any]:
    """Evaluate each baseline against one recorded model identifier at a time."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    baselines = baseline_result.get("baselines")
    if not isinstance(baselines, Mapping) or not baselines:
        raise ValueError("baseline result requires baselines")

    output: dict[str, Any] = {}
    shared_human_ids: set[str] | None = None
    all_models: set[str] = set()
    for baseline_name, baseline in baselines.items():
        rows = baseline.get("groups", [])
        human_rows = [row for row in rows if row.get("source") == "human"]
        human_ids = {str(row["group_id"]) for row in human_rows}
        if shared_human_ids is None:
            shared_human_ids = human_ids
        elif human_ids != shared_human_ids:
            raise ValueError("baselines do not share the same historical-reference test groups")
        models = sorted({str(row["model_id"]) for row in rows if row.get("source") == "ai"})
        all_models.update(models)
        by_model: dict[str, Any] = {}
        for model_id in models:
            model_rows = human_rows + [
                row
                for row in rows
                if row.get("source") == "ai" and str(row.get("model_id")) == model_id
            ]
            model_seed = _stable_seed(seed, baseline_name, model_id)
            by_model[model_id] = _stratified_bootstrap_auc(
                model_rows,
                n_resamples=n_resamples,
                seed=model_seed,
                min_groups_for_ci=min_groups_for_ci,
            )
        point_estimates = {
            model_id: float(result["point_estimate"]) for model_id, result in by_model.items()
        }
        lowest = min(point_estimates, key=point_estimates.get)
        highest = max(point_estimates, key=point_estimates.get)
        output[str(baseline_name)] = {
            "by_model": by_model,
            "across_model": {
                **_numeric_summary(list(point_estimates.values())),
                "lowest_model_id": lowest,
                "lowest_auroc": point_estimates[lowest],
                "highest_model_id": highest,
                "highest_auroc": point_estimates[highest],
            },
        }
    return {
        "n_recorded_model_identifiers": len(all_models),
        "recorded_model_identifiers": sorted(all_models),
        "n_shared_historical_test_groups": len(shared_human_ids or set()),
        "n_bootstrap_resamples": n_resamples,
        "bootstrap_unit": "source_or_generation_group_within_provenance_class",
        "endpoint": "equal_page_type_macro_auroc",
        "baselines": output,
        "interpretation": (
            "Model identifiers are provider-recorded strings, not independently verified model implementations. "
            "Every stratum reuses the same small historical test set and averages within-type AUROCs, so intervals "
            "are descriptive pilot diagnostics."
        ),
    }


def evaluate_challenge_composition_sensitivity(
    baseline_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the primary endpoint with alternative challenge-set weightings."""
    baselines = baseline_result.get("baselines")
    if not isinstance(baselines, Mapping) or not baselines:
        raise ValueError("baseline result requires baselines")

    output: dict[str, Any] = {}
    shared_models: set[str] | None = None
    for baseline_name, baseline in baselines.items():
        rows = baseline.get("groups")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"baseline {baseline_name!r} requires group rows")
        human_rows = [row for row in rows if row.get("source") == "human"]
        ai_rows = [row for row in rows if row.get("source") == "ai"]
        models = sorted({str(row.get("model_id")) for row in ai_rows})
        if shared_models is None:
            shared_models = set(models)
        elif set(models) != shared_models:
            raise ValueError("baselines do not share recorded model identifiers")

        def type_macro_point(subset: Sequence[Mapping[str, Any]]) -> float:
            return float(
                _stratified_bootstrap_auc(
                    subset,
                    n_resamples=1,
                    seed=0,
                    min_groups_for_ci=10**9,
                )["point_estimate"]
            )

        by_model = {
            model_id: type_macro_point(
                human_rows
                + [row for row in ai_rows if str(row.get("model_id")) == model_id]
            )
            for model_id in models
        }
        leave_one_model_out = (
            {
                model_id: type_macro_point(
                    human_rows
                    + [row for row in ai_rows if str(row.get("model_id")) != model_id]
                )
                for model_id in models
            }
            if len(models) > 1
            else {}
        )
        primary = float(baseline["type_macro"]["point_estimate"])
        pooled = float(baseline["overall"]["point_estimate"])
        output[str(baseline_name)] = {
            "equal_page_type_macro_auroc": primary,
            "pair_weighted_within_type_auroc": float(
                baseline["type_macro"]["pair_weighted_within_type_point_estimate"]
            ),
            "pooled_cross_type_auroc": pooled,
            "type_macro_minus_pooled": primary - pooled,
            "equal_model_mean_of_type_macro_aurocs": statistics.fmean(by_model.values()),
            "by_model_type_macro_auroc": by_model,
            "leave_one_model_out_type_macro_auroc": {
                "by_excluded_model": leave_one_model_out,
                "summary": (
                    _numeric_summary(list(leave_one_model_out.values()))
                    if leave_one_model_out
                    else {
                        "status": "not_applicable",
                        "reason": "leave-one-model-out requires at least two model strata",
                    }
                ),
            },
        }
    return {
        "design": "challenge_composition_reweighting_on_frozen_scores",
        "n_recorded_model_identifiers": len(shared_models or set()),
        "primary_endpoint": "equal_page_type_macro_auroc",
        "baselines": output,
        "interpretation": (
            "The page-type macro endpoint avoids unvalidated cross-type score comparisons. Equal-model and "
            "leave-one-model-out summaries test sensitivity to the recorded challenge mixture; they do not "
            "independently verify provider model identities."
        ),
    }


def _summarize_contributions(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not groups:
        raise ValueError("contribution summary requires at least one group")
    shares_by_dimension: dict[str, list[float]] = {dimension: [] for dimension in FEATURE_NAMES}
    contribution_totals: Counter[str] = Counter()
    dominant_counts: Counter[str] = Counter()
    max_shares: list[float] = []
    zero_total_groups = 0
    total_score = 0.0
    for group in groups:
        raw = group.get("per_dimension")
        if not isinstance(raw, Mapping):
            raise ValueError("scored group requires per_dimension contributions")
        contributions = {dimension: float(raw.get(dimension, 0.0)) for dimension in FEATURE_NAMES}
        if any(value < 0 for value in contributions.values()):
            raise ValueError("dimension contributions must be non-negative")
        total = sum(contributions.values())
        total_score += total
        contribution_totals.update(contributions)
        if total <= 0:
            zero_total_groups += 1
            for dimension in FEATURE_NAMES:
                shares_by_dimension[dimension].append(0.0)
            max_shares.append(0.0)
            continue
        shares = {dimension: value / total for dimension, value in contributions.items()}
        for dimension, share in shares.items():
            shares_by_dimension[dimension].append(share)
        dominant = max(FEATURE_NAMES, key=lambda dimension: (shares[dimension], dimension))
        dominant_counts[dominant] += 1
        max_shares.append(shares[dominant])
    n_groups = len(groups)
    return {
        "n_groups": n_groups,
        "zero_total_groups": zero_total_groups,
        "dominant_dimension": {
            dimension: {
                "count": dominant_counts[dimension],
                "fraction": dominant_counts[dimension] / n_groups,
            }
            for dimension in FEATURE_NAMES
        },
        "group_share_summary": {
            dimension: _numeric_summary(shares) for dimension, shares in shares_by_dimension.items()
        },
        "score_weighted_share": {
            dimension: (contribution_totals[dimension] / total_score if total_score > 0 else 0.0)
            for dimension in FEATURE_NAMES
        },
        "dominant_share_summary": _numeric_summary(max_shares),
    }


def _feature_ablation_aurocs(
    groups: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    min_groups_for_ci: int,
) -> dict[str, Any]:
    variants: dict[str, list[dict[str, Any]]] = {
        "full_profile": [],
        **{f"without_{dimension}": [] for dimension in FEATURE_NAMES},
        **{f"only_{dimension}": [] for dimension in FEATURE_NAMES},
        "without_color_and_spacing": [],
        "only_color_and_spacing": [],
    }
    ordered_groups = sorted(
        groups,
        key=lambda group: (str(group.get("source", "")), str(group.get("group_id", ""))),
    )
    for group in ordered_groups:
        contributions = {
            dimension: float(group["per_dimension"].get(dimension, 0.0))
            for dimension in FEATURE_NAMES
        }
        total = sum(contributions.values())
        common = {
            "source": str(group["source"]),
            "group_id": str(group.get("group_id", "")),
            "page_type": str(group.get("page_type", "")),
        }
        variants["full_profile"].append({**common, "reference_fit_score": -total})
        for dimension in FEATURE_NAMES:
            variants[f"without_{dimension}"].append({
                **common,
                "reference_fit_score": -(total - contributions[dimension]),
            })
            variants[f"only_{dimension}"].append({
                **common,
                "reference_fit_score": -contributions[dimension],
            })
        color_spacing = contributions["color"] + contributions["spacing"]
        variants["without_color_and_spacing"].append({
            **common,
            "reference_fit_score": -(total - color_spacing),
        })
        variants["only_color_and_spacing"].append({
            **common,
            "reference_fit_score": -color_spacing,
        })
    return {
        name: _stratified_bootstrap_auc(
            rows,
            n_resamples=n_resamples,
            seed=seed,
            min_groups_for_ci=min_groups_for_ci,
        )
        for name, rows in variants.items()
    }


def evaluate_feature_dominance(
    scored_result: Mapping[str, Any],
    *,
    n_resamples: int = 2_000,
    seed: int = 2027,
    min_groups_for_ci: int = 10,
) -> dict[str, Any]:
    """Summarize which released profile dimensions dominate each test score."""
    test = scored_result.get("test")
    if not isinstance(test, Mapping):
        raise ValueError("scored result requires a test object")
    groups = test.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("scored result requires non-empty test groups")
    by_source = {
        source: _summarize_contributions(
            [group for group in groups if str(group.get("source")) == source]
        )
        for source in sorted({str(group.get("source")) for group in groups})
    }
    by_page_type = {
        page_type: _summarize_contributions(
            [group for group in groups if str(group.get("page_type")) == page_type]
        )
        for page_type in sorted({str(group.get("page_type")) for group in groups})
    }
    return {
        "scorer": "released_profile_l2_w1",
        "ablation_endpoint": "equal_page_type_macro_auroc",
        "overall": _summarize_contributions(groups),
        "by_source": by_source,
        "by_page_type": by_page_type,
        "feature_ablation_auroc": _feature_ablation_aurocs(
            groups,
            n_resamples=n_resamples,
            seed=seed,
            min_groups_for_ci=min_groups_for_ci,
        ),
        "interpretation": (
            "Dominance identifies which normalized low-level proxy contributes most to this released distance. "
            "Single-feature and leave-one-feature-out AUROCs are sensitivity checks, not evidence that a "
            "dimension is causal, perceptually important, or defective."
        ),
    }


def summarize_viewport_stability(design_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize paired 390x844 versus 430x932 profile deltas."""
    paired = design_profile.get("paired_mobile_viewport_deltas")
    if not isinstance(paired, Mapping):
        raise ValueError("design profile requires paired_mobile_viewport_deltas")
    absolute = paired.get("absolute_delta")
    if not isinstance(absolute, Mapping) or not absolute:
        raise ValueError("paired viewport deltas require absolute_delta metrics")
    medians = {metric: float(summary["median"]) for metric, summary in absolute.items()}
    zero_metrics = sorted(metric for metric, median in medians.items() if abs(median) <= 1e-12)
    nonzero_metrics = sorted(set(medians).difference(zero_metrics))
    largest_metric = max(medians, key=medians.get)
    return {
        "paired_source_groups": int(paired["paired_source_groups"]),
        "viewports": sorted(design_profile.get("by_viewport", {}).keys()),
        "n_metrics": len(medians),
        "n_zero_median_absolute_delta": len(zero_metrics),
        "zero_median_metrics": zero_metrics,
        "nonzero_median_metrics": nonzero_metrics,
        "largest_median_absolute_delta": {
            "metric": largest_metric,
            "value": medians[largest_metric],
        },
        "absolute_delta": absolute,
        "interpretation": (
            "This paired summary covers two nearby mobile viewports only. It does not establish desktop, "
            "interaction-state, or cross-browser stability."
        ),
    }


def evaluate_measurement_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Quantify two extractor-level confounds visible in the frozen feature arrays."""
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for record in manifest.get("records", []):
        group_id = str(record.get("group_id", ""))
        if not group_id or group_id in seen_groups:
            raise ValueError("measurement audit requires exactly one scoring record per group")
        seen_groups.add(group_id)
        features = record.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"scoring record {group_id!r} requires feature arrays")
        samples = {dimension: list(features.get(dimension, [])) for dimension in FEATURE_NAMES}
        if any(not values for values in samples.values()):
            raise ValueError(f"scoring record {group_id!r} has an empty feature family")
        spacing = [float(value) for value in samples["spacing"]]
        colors = [str(value).casefold() for value in samples["color"] if str(value)]
        rows.append({
            "group_id": group_id,
            "source": str(record.get("source")),
            "page_type": str(record.get("page_type")),
            "sample_counts": {
                dimension: len(values) for dimension, values in samples.items()
            },
            "spacing_proxy_1_2_count": sum(abs(value - 1.2) <= 1e-9 for value in spacing),
            "spacing_count": len(spacing),
            "spacing_proxy_1_2_fraction": (
                sum(abs(value - 1.2) <= 1e-9 for value in spacing) / len(spacing)
            ),
            "color_unique_count": len(set(colors)),
        })

    source_names = {"human": "historical_reference", "ai": "generated_challenge"}
    unknown_sources = sorted({row["source"] for row in rows}.difference(source_names))
    if unknown_sources:
        raise ValueError(f"unexpected source labels in measurement audit: {unknown_sources}")

    by_cohort: dict[str, Any] = {}
    for source, output_name in source_names.items():
        cohort = [row for row in rows if row["source"] == source]
        if not cohort:
            raise ValueError(f"measurement audit has no {source!r} rows")
        color_unique = [float(row["color_unique_count"]) for row in cohort]
        color_samples = [float(row["sample_counts"]["color"]) for row in cohort]
        proxy_count = sum(int(row["spacing_proxy_1_2_count"]) for row in cohort)
        spacing_count = sum(int(row["spacing_count"]) for row in cohort)
        by_cohort[output_name] = {
            "n_groups": len(cohort),
            "sample_count_summary": {
                dimension: _numeric_summary(
                    [float(row["sample_counts"][dimension]) for row in cohort]
                )
                for dimension in FEATURE_NAMES
            },
            "spacing_value_1_2": {
                "pooled_sample_fraction": proxy_count / spacing_count,
                "per_group_fraction_summary": _numeric_summary(
                    [float(row["spacing_proxy_1_2_fraction"]) for row in cohort]
                ),
                "note": (
                    "The extractor maps unresolved CSS line-height:normal to 1.2, but an exact "
                    "1.2 value may also have been authored explicitly."
                ),
            },
            "color_unique_count_summary": _numeric_summary(color_unique),
            "color_unique_vs_color_sample_count_spearman": _spearman_correlation(
                color_unique, color_samples
            ),
        }
    return {
        "design": "frozen_feature_array_measurement_audit",
        "by_cohort": by_cohort,
        "interpretation": (
            "This audit measures extractor saturation and association with sample density. It cannot "
            "identify whether a 1.2 spacing value came from a browser proxy or explicit CSS, and a "
            "correlation between palette cardinality and sample count is not a causal estimate."
        ),
    }


def audit_ai_exclusions(
    manifest: Mapping[str, Any],
    exclusion_ledger: Mapping[str, Any],
    source_run_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that the 590/600 challenge subset follows a score-independent rule."""
    manifest_ai = [record for record in manifest.get("records", []) if record.get("source") == "ai"]
    exclusions = exclusion_ledger.get("exclusions")
    planned_rows = source_run_ledger.get("records")
    if exclusion_ledger.get("schema") != "webhumanbench_v1_ai_exclusion_ledger_v1":
        raise ValueError("unexpected AI exclusion ledger schema")
    if source_run_ledger.get("schema") != "webhumanbench_v1_ai_source_run_ledger_v1":
        raise ValueError("unexpected AI source-run ledger schema")
    if not isinstance(exclusions, list):
        raise ValueError("AI exclusion ledger requires an exclusions list")
    if not isinstance(planned_rows, list) or not planned_rows:
        raise ValueError("AI source-run ledger requires non-empty records")

    planned = {str(row["id"]): row for row in planned_rows}
    if len(planned) != len(planned_rows):
        raise ValueError("AI source-run ledger contains duplicate page ids")
    retained_page_ids = {
        str(record.get("provenance", {}).get("prompt_id")) for record in manifest_ai
    }
    excluded_page_ids = {str(row["page_id"]) for row in exclusions}
    if "None" in retained_page_ids:
        raise ValueError("benchmark AI record is missing provenance.prompt_id")
    if len(excluded_page_ids) != len(exclusions):
        raise ValueError("AI exclusion ledger contains duplicate page ids")
    if retained_page_ids.intersection(excluded_page_ids):
        raise ValueError("an AI page is both retained and excluded")
    if retained_page_ids.union(excluded_page_ids) != set(planned):
        raise ValueError("retained and excluded AI pages do not close the 600-page source run")
    if int(source_run_ledger.get("planned_pages", -1)) != len(planned):
        raise ValueError("AI source-run ledger planned_pages does not match its records")
    if int(exclusion_ledger.get("retained_pages", -1)) != len(retained_page_ids):
        raise ValueError("AI exclusion ledger retained_pages does not match the manifest")
    if int(exclusion_ledger.get("excluded_pages", -1)) != len(excluded_page_ids):
        raise ValueError("AI exclusion ledger excluded_pages does not match its rows")
    if exclusion_ledger.get("selection_uses_reference_fit_score") is not False:
        raise ValueError("AI exclusion ledger must declare score-independent selection")

    details: list[dict[str, Any]] = []
    for exclusion in sorted(exclusions, key=lambda row: str(row["page_id"])):
        page_id = str(exclusion["page_id"])
        error = str(exclusion.get("error", ""))
        if "requested resources outside its archived HTML" not in error:
            raise ValueError(f"AI exclusion {page_id!r} is not an offline-closure failure")
        source = planned[page_id]
        if str(exclusion.get("page_type")) != str(source["page_type"]):
            raise ValueError(f"AI exclusion {page_id!r} has inconsistent page_type")
        if str(exclusion.get("model_id")) != str(source["model_id"]):
            raise ValueError(f"AI exclusion {page_id!r} has inconsistent model_id")
        details.append({
            "page_id": page_id,
            "page_type": str(source["page_type"]),
            "model_id": str(source["model_id"]),
            "reason_code": "external_network_dependency",
            "evidence": error,
        })

    page_types = sorted({str(row["page_type"]) for row in planned_rows})
    by_page_type = {}
    for page_type in page_types:
        planned_ids = {
            str(row["id"]) for row in planned_rows if str(row["page_type"]) == page_type
        }
        excluded_ids = {
            detail["page_id"] for detail in details if detail["page_type"] == page_type
        }
        by_page_type[page_type] = {
            "planned": len(planned_ids),
            "retained": len(planned_ids.difference(excluded_ids)),
            "excluded": len(excluded_ids),
        }
    models = sorted({str(row["model_id"]) for row in planned_rows})
    by_model = {}
    for model_id in models:
        planned_ids = {
            str(row["id"]) for row in planned_rows if str(row["model_id"]) == model_id
        }
        excluded_ids = {
            detail["page_id"] for detail in details if detail["model_id"] == model_id
        }
        by_model[model_id] = {
            "planned": len(planned_ids),
            "retained": len(planned_ids.difference(excluded_ids)),
            "excluded": len(excluded_ids),
        }
    return {
        "source_run_rows": len(planned),
        "retained_rows": len(retained_page_ids),
        "excluded_rows": len(excluded_page_ids),
        "selection_rule": "offline_local_render_with_complete_feature_arrays_and_no_external_network_dependency",
        "selection_uses_reference_fit_score": False,
        "closure_verified": True,
        "by_page_type": by_page_type,
        "by_model": by_model,
        "exclusions": details,
        "interpretation": (
            "The ten rows are omitted because their archived HTML requests external placeholder assets. "
            "The rule is fixed before reference-fit scoring, so the 590-row subset is not score-selected."
        ),
    }


def build_v1_diagnostics(
    manifest: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    scored_result: Mapping[str, Any],
    design_profile: Mapping[str, Any],
    ai_exclusion_ledger: Mapping[str, Any],
    ai_source_run_ledger: Mapping[str, Any],
    *,
    n_alternative_splits: int = 100,
    n_bootstrap: int = 2_000,
    seed: int = 2027,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Build the complete frozen-data diagnostic artifact used by the paper."""
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "benchmark": manifest.get("metadata", {}).get("benchmark_name"),
        "version": manifest.get("metadata", {}).get("version"),
        "seed": seed,
        "seeds": {
            "source_split": seed,
            "group_bootstrap": bootstrap_seed,
        },
        "source_split_sensitivity": evaluate_split_sensitivity(
            manifest,
            n_alternative_splits=n_alternative_splits,
            seed=seed,
        ),
        "leave_one_reference_out_coverage": evaluate_leave_one_reference_out(
            manifest,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "model_stratified_diagnostics": evaluate_model_strata(
            baseline_result,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "challenge_composition_sensitivity": evaluate_challenge_composition_sensitivity(
            baseline_result
        ),
        "feature_contribution_dominance": evaluate_feature_dominance(
            scored_result,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "paired_mobile_viewport_stability": summarize_viewport_stability(design_profile),
        "measurement_artifact_diagnostics": evaluate_measurement_artifacts(manifest),
        "ai_exclusion_audit": audit_ai_exclusions(
            manifest, ai_exclusion_ledger, ai_source_run_ledger
        ),
        "claim_boundary": (
            "All diagnostics are conditional on the frozen v1 construction. They do not resolve temporal, "
            "content-brief, source-selection, cultural, browser, or verified-authorship confounding."
        ),
    }
