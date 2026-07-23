"""Reference-only baselines for WebHumanBench design-fit diagnostics.

The benchmark does not train an authorship detector.  These baselines instead
fit only the historical-reference ``train`` groups and rank held-out pages by
their distance from that declared reference.  Human-versus-AI AUROC is retained
as a corpus-conditional provenance diagnostic, never as an authorship claim.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .bias import BiasScorer, wasserstein1_distance
from .features import FEATURE_NAMES, PageFeatures
from .human_likeness import (
    BenchmarkRecord,
    fit_reference_by_page_type,
    roc_auc_human_vs_ai,
    validate_manifest,
)

BASELINE_SCHEMA = "webmark_reference_fit_baselines_v2"
BASELINE_NAMES = (
    "profile_l2_w1",
    "diagonal_l2",
    "nearest_train_l1",
    "robust_mad_l1",
    "typography_w1",
)
SCALE_EPSILON = 1e-9


def _page_scalar_features(features: PageFeatures) -> dict[str, float]:
    """Return the five scalar summaries used by simple reference baselines."""
    values: dict[str, float] = {}
    for dimension in FEATURE_NAMES:
        samples = getattr(features, dimension)
        if dimension == "color":
            values[dimension] = float(len({str(value).lower() for value in samples if value}))
        else:
            values[dimension] = statistics.fmean(float(value) for value in samples)
    return values


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _group_rows(records: Sequence[BenchmarkRecord]) -> list[dict[str, Any]]:
    """Aggregate viewport records without changing the source-group endpoint."""
    grouped: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)

    rows: list[dict[str, Any]] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        first = members[0]
        if any(
            (record.source, record.page_type, record.model_id, record.split)
            != (first.source, first.page_type, first.model_id, first.split)
            for record in members
        ):
            raise ValueError(f"group {group_id!r} mixes provenance, page type, model, or split")
        page_values = [_page_scalar_features(record.features) for record in members]
        values = {
            dimension: statistics.fmean(page[dimension] for page in page_values)
            for dimension in FEATURE_NAMES
        }
        typography = [sample for record in members for sample in record.features.typography]
        if not typography:
            raise ValueError(f"group {group_id!r} has no typography samples")
        rows.append({
            "group_id": group_id,
            "source": first.source,
            "split": first.split,
            "page_type": first.page_type,
            "model_id": first.model_id,
            "n_records": len(members),
            "values": values,
            "typography": typography,
        })
    return rows


def _safe_scale(values: Sequence[float]) -> float:
    if len(values) > 1:
        scale = statistics.stdev(values)
        if scale > SCALE_EPSILON:
            return scale
    return 0.0


def _reference_parameters(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(train_rows) < 2:
        raise ValueError("reference baselines require at least two train source groups")
    values_by_dimension = {
        dimension: [float(row["values"][dimension]) for row in train_rows]
        for dimension in FEATURE_NAMES
    }
    medians = {dimension: statistics.median(values) for dimension, values in values_by_dimension.items()}
    mad_scales: dict[str, float] = {}
    for dimension, values in values_by_dimension.items():
        median = medians[dimension]
        mad = statistics.median(abs(value - median) for value in values)
        robust_scale = 1.4826 * mad
        fallback = _safe_scale(values)
        mad_scales[dimension] = robust_scale if robust_scale > SCALE_EPSILON else fallback
    stds = {dimension: _safe_scale(values) for dimension, values in values_by_dimension.items()}
    active_dimensions = [
        dimension for dimension in FEATURE_NAMES if stds[dimension] > SCALE_EPSILON
    ]
    return {
        "means": {dimension: statistics.fmean(values) for dimension, values in values_by_dimension.items()},
        "stds": stds,
        "medians": medians,
        "mad_scales": mad_scales,
        "active_dimensions": active_dimensions,
        "inactive_zero_variance_dimensions": [
            dimension for dimension in FEATURE_NAMES if dimension not in active_dimensions
        ],
        "train_vectors": [dict(row["values"]) for row in train_rows],
        "typography": [sample for row in train_rows for sample in row["typography"]],
    }


def _score_simple_baseline(
    baseline: str,
    row: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> float:
    values = row["values"]
    means = parameters["means"]
    stds = parameters["stds"]
    active_dimensions = parameters["active_dimensions"]
    if baseline == "diagonal_l2":
        return sum(
            ((float(values[dimension]) - float(means[dimension])) / float(stds[dimension])) ** 2
            for dimension in active_dimensions
        )
    if baseline == "nearest_train_l1":
        return min(
            sum(
                abs(float(values[dimension]) - float(train_vector[dimension])) / float(stds[dimension])
                for dimension in active_dimensions
            )
            for train_vector in parameters["train_vectors"]
        )
    if baseline == "robust_mad_l1":
        medians = parameters["medians"]
        scales = parameters["mad_scales"]
        return sum(
            abs(float(values[dimension]) - float(medians[dimension])) / float(scales[dimension])
            for dimension in active_dimensions
        )
    if baseline == "typography_w1":
        if "typography" not in active_dimensions:
            return 0.0
        return wasserstein1_distance(row["typography"], parameters["typography"]) / float(stds["typography"])
    raise ValueError(f"unsupported simple baseline {baseline!r}")


def _profile_scores(
    train_records: Sequence[BenchmarkRecord], test_records: Sequence[BenchmarkRecord]
) -> dict[str, float]:
    """Reproduce the released L2+W1 profile score at the group endpoint."""
    references = fit_reference_by_page_type(train_records)
    scorers = {
        page_type: BiasScorer(reference, penalty="l2", nonparametric_dims=("typography",))
        for page_type, reference in references.items()
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in test_records:
        grouped[record.group_id].append(scorers[record.page_type].score(record.features).total)
    return {group_id: statistics.fmean(scores) for group_id, scores in grouped.items()}


def _bootstrap_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    min_groups_for_ci: int,
) -> dict[str, Any]:
    human = [float(row["reference_fit_score"]) for row in rows if row["source"] == "human"]
    ai = [float(row["reference_fit_score"]) for row in rows if row["source"] == "ai"]
    point = roc_auc_human_vs_ai(rows, score_field="reference_fit_score")
    output: dict[str, Any] = {
        "point_estimate": point,
        "n_human_groups": len(human),
        "n_ai_groups": len(ai),
        "n_resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "source_or_generation_group",
    }
    if min(len(human), len(ai)) < min_groups_for_ci:
        output.update({
            "status": "not_reportable",
            "reason": (
                f"At least {min_groups_for_ci} groups are required in each provenance class for a bootstrap CI; "
                f"observed human={len(human)}, ai={len(ai)}."
            ),
        })
        return output
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        sampled_human = [rng.choice(human) for _ in human]
        sampled_ai = [rng.choice(ai) for _ in ai]
        sample_rows = (
            [{"source": "human", "reference_fit_score": value} for value in sampled_human]
            + [{"source": "ai", "reference_fit_score": value} for value in sampled_ai]
        )
        samples.append(roc_auc_human_vs_ai(sample_rows, score_field="reference_fit_score"))
    output.update({
        "status": "reportable",
        "ci_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
    })
    return output


def _stratified_bootstrap_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    min_groups_for_ci: int,
) -> dict[str, Any]:
    """Estimate equal-page-type macro AUROC with within-type resampling.

    Reference profiles are fitted independently by page type, so their raw
    distances are not assumed to be comparable across types. The primary
    endpoint therefore computes an AUROC inside each page type and gives each
    declared type equal weight.
    """
    page_types = sorted({str(row["page_type"]) for row in rows})
    if not page_types:
        raise ValueError("stratified AUROC requires at least one page type")

    strata: dict[str, dict[str, list[float]]] = {}
    per_page_type: dict[str, dict[str, float | int]] = {}
    for page_type in page_types:
        type_rows = [row for row in rows if str(row["page_type"]) == page_type]
        human = [
            float(row["reference_fit_score"])
            for row in type_rows
            if row["source"] == "human"
        ]
        ai = [
            float(row["reference_fit_score"])
            for row in type_rows
            if row["source"] == "ai"
        ]
        if not human or not ai:
            raise ValueError(
                f"stratified AUROC requires both provenance classes for {page_type!r}"
            )
        point = roc_auc_human_vs_ai(type_rows, score_field="reference_fit_score")
        strata[page_type] = {"human": human, "ai": ai}
        per_page_type[page_type] = {
            "point_estimate": point,
            "n_human_groups": len(human),
            "n_ai_groups": len(ai),
            "n_pairs": len(human) * len(ai),
        }

    type_points = [float(row["point_estimate"]) for row in per_page_type.values()]
    pair_count = sum(int(row["n_pairs"]) for row in per_page_type.values())
    pair_weighted = sum(
        float(row["point_estimate"]) * int(row["n_pairs"])
        for row in per_page_type.values()
    ) / pair_count
    total_human = sum(len(stratum["human"]) for stratum in strata.values())
    total_ai = sum(len(stratum["ai"]) for stratum in strata.values())
    output: dict[str, Any] = {
        "point_estimate": statistics.fmean(type_points),
        "pair_weighted_within_type_point_estimate": pair_weighted,
        "n_page_types": len(page_types),
        "n_human_groups": total_human,
        "n_ai_groups": total_ai,
        "n_resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "source_or_generation_group_within_page_type_and_provenance_class",
        "aggregation": "equal_weight_mean_of_page_type_aurocs",
        "by_page_type": per_page_type,
    }
    if min(total_human, total_ai) < min_groups_for_ci:
        output.update({
            "status": "not_reportable",
            "reason": (
                f"At least {min_groups_for_ci} total groups are required in each provenance class "
                f"for a stratified bootstrap CI; observed human={total_human}, ai={total_ai}."
            ),
        })
        return output

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        type_samples: list[float] = []
        for page_type in page_types:
            human = strata[page_type]["human"]
            ai = strata[page_type]["ai"]
            sampled_rows = (
                [
                    {
                        "source": "human",
                        "reference_fit_score": rng.choice(human),
                    }
                    for _ in human
                ]
                + [
                    {
                        "source": "ai",
                        "reference_fit_score": rng.choice(ai),
                    }
                    for _ in ai
                ]
            )
            type_samples.append(
                roc_auc_human_vs_ai(sampled_rows, score_field="reference_fit_score")
            )
        samples.append(statistics.fmean(type_samples))
    output.update({
        "status": "reportable",
        "ci_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "bootstrap_boundary": (
            "Conditional on the frozen test sources; strata with one historical test source "
            "cannot express between-source uncertainty."
        ),
    })
    return output


def _summarize_baseline(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    min_groups_for_ci: int,
) -> dict[str, Any]:
    by_page_type: dict[str, Any] = {}
    for page_type in sorted({str(row["page_type"]) for row in rows}):
        page_rows = [row for row in rows if row["page_type"] == page_type]
        by_page_type[page_type] = _bootstrap_auc(
            page_rows,
            n_resamples=n_resamples,
            seed=seed,
            min_groups_for_ci=min_groups_for_ci,
        )
    return {
        "name": name,
        "distance_direction": "lower_is_closer_to_the_declared_reference",
        "type_macro": _stratified_bootstrap_auc(
            rows,
            n_resamples=n_resamples,
            seed=seed,
            min_groups_for_ci=min_groups_for_ci,
        ),
        "overall": _bootstrap_auc(
            rows,
            n_resamples=n_resamples,
            seed=seed,
            min_groups_for_ci=min_groups_for_ci,
        ),
        "by_page_type": by_page_type,
        "groups": list(rows),
    }


def evaluate_reference_fit_baselines(
    manifest: Mapping[str, Any],
    *,
    n_resamples: int = 2_000,
    seed: int = 42,
    min_groups_for_ci: int = 10,
) -> dict[str, Any]:
    """Evaluate fixed, train-only reference baselines on the held-out test split."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if min_groups_for_ci <= 0:
        raise ValueError("min_groups_for_ci must be positive")
    records = validate_manifest(manifest)
    train_records = [record for record in records if record.split == "train"]
    test_records = [record for record in records if record.split == "test"]
    train_groups = _group_rows(train_records)
    test_groups = _group_rows(test_records)
    train_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_groups:
        train_by_type[str(row["page_type"])].append(row)
    parameters_by_type = {
        page_type: _reference_parameters(rows) for page_type, rows in train_by_type.items()
    }
    profile_scores = _profile_scores(train_records, test_records)

    summaries: dict[str, dict[str, Any]] = {}
    for baseline in BASELINE_NAMES:
        rows: list[dict[str, Any]] = []
        for group in test_groups:
            if baseline == "profile_l2_w1":
                distance = profile_scores[str(group["group_id"])]
            else:
                distance = _score_simple_baseline(
                    baseline,
                    group,
                    parameters_by_type[str(group["page_type"])],
                )
            rows.append({
                "group_id": group["group_id"],
                "source": group["source"],
                "page_type": group["page_type"],
                "model_id": group["model_id"],
                "distance": distance,
                "reference_fit_score": -distance,
            })
        summaries[baseline] = _summarize_baseline(
            baseline,
            rows,
            n_resamples=n_resamples,
            seed=seed,
            min_groups_for_ci=min_groups_for_ci,
        )
    return {
        "schema": BASELINE_SCHEMA,
        "benchmark": manifest["metadata"]["benchmark_name"],
        "version": manifest["metadata"]["version"],
        "task": "reference_relative_provenance_separation_diagnostic",
        "primary_endpoint": "equal_page_type_macro_auroc",
        "compatibility_endpoint": "pooled_cross_type_auroc",
        "fit_split": "human_train_only",
        "evaluation_split": "group_disjoint_test",
        "n_train_human_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "reference_scale_diagnostics": {
            page_type: {
                "active_dimensions": parameters["active_dimensions"],
                "inactive_zero_variance_dimensions": parameters[
                    "inactive_zero_variance_dimensions"
                ],
            }
            for page_type, parameters in sorted(parameters_by_type.items())
        },
        "zero_variance_policy": (
            "A feature with zero variation across the train-only source groups in a page type is excluded "
            "from normalized baseline distances; this is an insufficient-variation flag, not an exact-match rule."
        ),
        "baselines": summaries,
        "interpretation": (
            "All methods rank distance to the declared historical-reference distribution. The primary "
            "endpoint averages within-page-type AUROCs because independently fitted type-specific distances "
            "are not assumed to be comparable across types. The pooled cross-type result is retained only for "
            "compatibility. Neither endpoint is a human-authorship detector, quality metric, or preference result."
        ),
    }
