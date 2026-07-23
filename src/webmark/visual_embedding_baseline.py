"""One-class visual-embedding baselines for WebHumanBench.

The module consumes frozen image embeddings and benchmark group metadata. It
does not load a model or treat an embedding as an aesthetic or authorship
label.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .benchmark_baselines import _stratified_bootstrap_auc
from .benchmark_diagnostics import (
    _alternative_assignment,
    _assignment_signature,
    _frozen_assignments,
    _human_groups_by_type,
    _numeric_summary,
    _sourcewise_coverage,
    _split_counts_by_type,
    _summarize_crossfit_coverage,
)

VISUAL_BASELINE_SCHEMA = "webmark_visual_embedding_baseline_v1"
BASELINES = ("embedding_centroid_cosine", "embedding_nearest_cosine")


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite numeric values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        raise ValueError("embedding norm must be positive")
    return tuple(value / norm for value in values)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    similarity = sum(a * b for a, b in zip(left, right, strict=True))
    return 1.0 - max(-1.0, min(1.0, similarity))


def _centroid(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("centroid requires at least one vector")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding dimensions do not match")
    return _normalize(
        [statistics.fmean(vector[index] for vector in vectors) for index in range(dimension)]
    )


def _distance(
    baseline: str,
    vector: Sequence[float],
    train_vectors: Sequence[Sequence[float]],
) -> float:
    if baseline == "embedding_centroid_cosine":
        return _cosine_distance(vector, _centroid(train_vectors))
    if baseline == "embedding_nearest_cosine":
        return min(_cosine_distance(vector, train) for train in train_vectors)
    raise ValueError(f"unsupported visual baseline {baseline!r}")


def _group_metadata(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in manifest.get("records", []):
        group_id = str(record["group_id"])
        row = {
            "group_id": group_id,
            "source": str(record["source"]),
            "split": str(record["split"]),
            "page_type": str(record["page_type"]),
            "model_id": record.get("model_id"),
        }
        prior = metadata.setdefault(group_id, row)
        if prior != row:
            raise ValueError(f"group {group_id!r} mixes benchmark metadata")
    if not metadata:
        raise ValueError("manifest contains no benchmark groups")
    return metadata


def _validated_embeddings(
    metadata: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, ...]]:
    if set(metadata) != set(embeddings):
        missing = sorted(set(metadata) - set(embeddings))
        extra = sorted(set(embeddings) - set(metadata))
        raise ValueError(
            f"embedding group IDs do not match the manifest; missing={missing[:3]}, extra={extra[:3]}"
        )
    normalized = {group_id: _normalize(embeddings[group_id]) for group_id in metadata}
    dimensions = {len(vector) for vector in normalized.values()}
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions are inconsistent")
    return normalized


def _score_assignment(
    metadata: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    assignments: Mapping[str, str],
    *,
    baseline: str,
) -> list[dict[str, Any]]:
    train_by_type: dict[str, list[Sequence[float]]] = defaultdict(list)
    for group_id, row in metadata.items():
        if row["source"] == "human" and assignments.get(group_id, row["split"]) == "train":
            train_by_type[str(row["page_type"])].append(embeddings[group_id])
    page_types = {str(row["page_type"]) for row in metadata.values()}
    if set(train_by_type) != page_types:
        raise ValueError("every page type requires historical-reference training embeddings")

    rows: list[dict[str, Any]] = []
    for group_id, row in metadata.items():
        split = assignments.get(group_id, str(row["split"]))
        if row["source"] == "human" and split != "test":
            continue
        if row["source"] != "human" and row["split"] != "test":
            continue
        distance = _distance(
            baseline,
            embeddings[group_id],
            train_by_type[str(row["page_type"])],
        )
        rows.append(
            {
                "group_id": group_id,
                "source": row["source"],
                "split": "test",
                "page_type": row["page_type"],
                "model_id": row["model_id"],
                "distance": distance,
                "reference_fit_score": -distance,
            }
        )
    return rows


def _split_sensitivity(
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    frozen: Mapping[str, float],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    groups_by_type = _human_groups_by_type(manifest)
    counts_by_type = _split_counts_by_type(manifest)
    frozen_assignments = _frozen_assignments(manifest)
    frozen_signature = _assignment_signature(frozen_assignments)
    signatures = {frozen_signature}
    runs: list[dict[str, Any]] = []
    max_attempts = max(10_000, n_splits * 200)
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
        aurocs = {}
        for baseline in BASELINES:
            rows = _score_assignment(
                metadata,
                embeddings,
                assignments,
                baseline=baseline,
            )
            endpoint = _stratified_bootstrap_auc(
                rows,
                n_resamples=1,
                seed=seed,
                min_groups_for_ci=10_000,
            )
            aurocs[baseline] = float(endpoint["point_estimate"])
        runs.append(
            {
                "index": len(runs) + 1,
                "assignment_sha256": signature,
                "aurocs": aurocs,
            }
        )
        if len(runs) == n_splits:
            break
    if len(runs) != n_splits:
        raise ValueError(f"could construct only {len(runs)} unique visual-baseline splits")
    summaries = {}
    for baseline in BASELINES:
        values = [float(run["aurocs"][baseline]) for run in runs]
        frozen_value = float(frozen[baseline])
        summaries[baseline] = {
            **_numeric_summary(values),
            "frozen_split": frozen_value,
            "frozen_percentile_among_alternatives": 100.0
            * (
                sum(value < frozen_value for value in values)
                + 0.5 * sum(value == frozen_value for value in values)
            )
            / len(values),
            "fraction_below_0_5": sum(value < 0.5 for value in values) / len(values),
        }
    return {
        "design": "deterministic_stratified_source_group_resplitting",
        "seed": seed,
        "n_alternative_splits": len(runs),
        "frozen_assignment_sha256": frozen_signature,
        "summary": summaries,
        "runs": runs,
    }


def _leave_one_reference_out(
    metadata: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    *,
    baseline: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    human_by_type: dict[str, list[str]] = defaultdict(list)
    challenge_by_type: dict[str, list[str]] = defaultdict(list)
    for group_id, row in metadata.items():
        target = human_by_type if row["source"] == "human" else challenge_by_type
        target[str(row["page_type"])].append(group_id)
    rows = []
    for page_type in sorted(human_by_type):
        humans = sorted(human_by_type[page_type])
        challenges = sorted(challenge_by_type[page_type])
        for heldout in humans:
            train_vectors = [embeddings[group_id] for group_id in humans if group_id != heldout]
            heldout_distance = _distance(baseline, embeddings[heldout], train_vectors)
            challenge_distances = [
                _distance(baseline, embeddings[group_id], train_vectors)
                for group_id in challenges
            ]
            rows.append(
                {
                    "group_id": heldout,
                    "page_type": page_type,
                    "n_fit_sources": len(train_vectors),
                    "n_challenge_groups": len(challenges),
                    "heldout_distance": heldout_distance,
                    "coverage_fraction": _sourcewise_coverage(
                        heldout_distance,
                        challenge_distances,
                    ),
                }
            )
    return {
        "summary": _summarize_crossfit_coverage(
            rows,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "sources": rows,
    }


def evaluate_visual_embedding_baselines(
    manifest: Mapping[str, Any],
    embeddings: Mapping[str, Sequence[float]],
    *,
    n_resamples: int = 2_000,
    n_alternative_splits: int = 100,
    seed: int = 2041,
) -> dict[str, Any]:
    """Evaluate frozen, resplit, and LORO image-embedding reference fit."""
    if n_resamples <= 0 or n_alternative_splits <= 0:
        raise ValueError("resample and split counts must be positive")
    metadata = _group_metadata(manifest)
    normalized = _validated_embeddings(metadata, embeddings)
    frozen_assignments = _frozen_assignments(manifest)
    baselines = {}
    frozen_points = {}
    for index, baseline in enumerate(BASELINES):
        rows = _score_assignment(
            metadata,
            normalized,
            frozen_assignments,
            baseline=baseline,
        )
        endpoint = _stratified_bootstrap_auc(
            rows,
            n_resamples=n_resamples,
            seed=seed + index,
            min_groups_for_ci=10,
        )
        frozen_points[baseline] = float(endpoint["point_estimate"])
        baselines[baseline] = {
            "distance_direction": "lower_is_closer_to_the_declared_reference",
            "frozen_type_macro": endpoint,
            "groups": rows,
            "leave_one_reference_out": _leave_one_reference_out(
                metadata,
                normalized,
                baseline=baseline,
                n_resamples=n_resamples,
                seed=seed + 100 + index,
            ),
        }
    split_sensitivity = _split_sensitivity(
        manifest,
        metadata,
        normalized,
        frozen_points,
        n_splits=n_alternative_splits,
        seed=2027,
    )
    for baseline in BASELINES:
        baselines[baseline]["source_split_sensitivity"] = split_sensitivity["summary"][baseline]
    return {
        "schema": VISUAL_BASELINE_SCHEMA,
        "task": "one_class_reference_relative_visual_embedding_fit",
        "embedding_dimension": len(next(iter(normalized.values()))),
        "n_groups": len(normalized),
        "baselines": baselines,
        "source_split_runs": split_sensitivity,
        "claim_boundary": (
            "The frozen image encoder supplies a representation, not a label. Results remain "
            "conditional on the same small historical-reference sources, screenshots, page types, "
            "and generated challenge cohort."
        ),
    }
