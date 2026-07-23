"""Scale and temporal-control validation for WebHumanBench.

The experiment expands the historical-reference corpus and introduces a
post-cutoff open-source control. It tests reference-profile stability and
whether AI/reference separation exceeds the temporal drift observed between
matched pre- and post-cutoff open-source pages. It does not infer authorship or
measure aesthetic preference.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .bias import BiasScorer
from .human_likeness import BenchmarkRecord, fit_reference_by_page_type, record_from_dict
from .release import canonical_json_sha256

PANEL_SCHEMA = "webmark_scale_temporal_panel_v1"
PROTOCOL_SCHEMA = "webmark_scale_temporal_protocol_v1"
REGISTRY_SCHEMA = "webmark_temporal_group_registry_v1"
RESULT_SCHEMA = "webmark_scale_temporal_validation_v1"

PRE_COHORT = "historical_pre_2023"
POST_COHORT = "contemporary_open_source"
AI_COHORT = "ai_generated"
COHORTS = (PRE_COHORT, POST_COHORT, AI_COHORT)


@dataclass(frozen=True)
class TemporalGroup:
    """Group-level temporal and matching metadata."""

    group_id: str
    leakage_group_id: str
    source: str
    split: str
    page_type: str
    cohort: str
    timestamp: datetime
    match_id: str | None
    matching: Mapping[str, str]


@dataclass(frozen=True)
class ValidatedPanel:
    """Validated records and group-level joins used by the experiment."""

    records: tuple[BenchmarkRecord, ...]
    groups: Mapping[str, TemporalGroup]
    match_blocks: Mapping[str, Mapping[str, str]]
    audit: Mapping[str, Any]


def _timestamp(value: Any, field: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} requires an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} requires an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "ci_95": [_percentile(values, 0.025), _percentile(values, 0.975)],
        "min": min(values),
        "max": max(values),
    }


def validate_scale_temporal_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the frozen design and its predeclared decision thresholds."""
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema must be {PROTOCOL_SCHEMA!r}")
    if not protocol.get("protocol_id"):
        raise ValueError("protocol requires protocol_id")
    _timestamp(protocol.get("temporal_cutoff"), "protocol.temporal_cutoff")
    page_types = protocol.get("required_page_types")
    if (
        not isinstance(page_types, list)
        or not page_types
        or len(page_types) != len(set(page_types))
    ):
        raise ValueError("protocol requires unique required_page_types")

    targets = protocol.get("targets_per_page_type")
    if not isinstance(targets, Mapping):
        raise ValueError("protocol requires targets_per_page_type")
    required_targets = {
        PRE_COHORT: ("train", "dev", "test"),
        POST_COHORT: ("test",),
        AI_COHORT: ("test",),
    }
    for cohort, splits in required_targets.items():
        cohort_targets = targets.get(cohort)
        if not isinstance(cohort_targets, Mapping):
            raise ValueError(f"protocol targets require cohort {cohort!r}")
        for split in splits:
            _positive_int(cohort_targets.get(split), f"targets.{cohort}.{split}")
    match_target = _positive_int(
        protocol.get("complete_temporal_match_blocks_per_page_type"),
        "complete_temporal_match_blocks_per_page_type",
    )
    if match_target > min(
        int(targets[PRE_COHORT]["test"]),
        int(targets[POST_COHORT]["test"]),
        int(targets[AI_COHORT]["test"]),
    ):
        raise ValueError("temporal match target exceeds a required test cohort")

    matching = protocol.get("matching")
    if not isinstance(matching, Mapping) or matching.get("frozen_before_scoring") is not True:
        raise ValueError("protocol matching must be frozen_before_scoring")
    exact_fields = matching.get("exact_fields")
    if not isinstance(exact_fields, list) or not exact_fields:
        raise ValueError("protocol matching requires exact_fields")
    prohibited = set(matching.get("prohibited_fields", []))
    if prohibited.intersection(exact_fields):
        raise ValueError("matching exact_fields may not contain prohibited fields")

    learning = protocol.get("learning_curve")
    if not isinstance(learning, Mapping):
        raise ValueError("protocol requires learning_curve")
    sizes = learning.get("train_groups_per_type")
    if (
        not isinstance(sizes, list)
        or not sizes
        or sizes != sorted(set(sizes))
        or any(not isinstance(size, int) or size < 2 for size in sizes)
    ):
        raise ValueError("learning_curve.train_groups_per_type must be unique sorted integers >= 2")
    _positive_int(learning.get("n_subsamples"), "learning_curve.n_subsamples")
    _positive_int(learning.get("seed"), "learning_curve.seed")
    stability_n = _positive_int(
        learning.get("stability_evaluation_train_groups_per_type"),
        "learning_curve.stability_evaluation_train_groups_per_type",
    )
    if stability_n not in sizes:
        raise ValueError("stability evaluation size must occur in the learning curve")
    if not 0 <= float(learning.get("min_spearman_vs_full", -1)) <= 1:
        raise ValueError("learning_curve.min_spearman_vs_full must be in [0, 1]")
    if float(learning.get("max_macro_auc_absolute_drift", -1)) < 0:
        raise ValueError("learning_curve.max_macro_auc_absolute_drift must be non-negative")

    temporal = protocol.get("temporal_validation")
    if not isinstance(temporal, Mapping):
        raise ValueError("protocol requires temporal_validation")
    _positive_int(temporal.get("n_bootstrap"), "temporal_validation.n_bootstrap")
    _positive_int(temporal.get("seed"), "temporal_validation.seed")
    if not 0.5 <= float(temporal.get("min_post_vs_ai_auc_ci_low", 0)) <= 1:
        raise ValueError("temporal_validation.min_post_vs_ai_auc_ci_low must be in [0.5, 1]")
    if float(temporal.get("min_ai_minus_post_distance_ci_low", -1)) < 0:
        raise ValueError(
            "temporal_validation.min_ai_minus_post_distance_ci_low must be non-negative"
        )


def _group_metadata(
    raw: Mapping[str, Any], record: BenchmarkRecord, cutoff: datetime, capture_sha256: str
) -> TemporalGroup:
    cohort = str(raw.get("temporal_cohort") or "")
    if cohort not in COHORTS:
        raise ValueError(f"record {record.id!r} has unsupported temporal_cohort {cohort!r}")
    expected_source = "ai" if cohort == AI_COHORT else "human"
    if record.source != expected_source:
        raise ValueError(
            f"record {record.id!r} cohort {cohort!r} requires source {expected_source!r}"
        )
    if cohort in (POST_COHORT, AI_COHORT) and record.split != "test":
        raise ValueError(f"record {record.id!r} cohort {cohort!r} is test-only")

    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"record {record.id!r} requires provenance")
    if provenance.get("capture_protocol_sha256") != capture_sha256:
        raise ValueError(f"record {record.id!r} does not bind the panel capture protocol")
    timestamp_field = "generated_at" if cohort == AI_COHORT else "revision_timestamp"
    observed_at = _timestamp(
        provenance.get(timestamp_field), f"record {record.id!r}.{timestamp_field}"
    )
    if cohort == PRE_COHORT and observed_at >= cutoff:
        raise ValueError(f"record {record.id!r} is not before the temporal cutoff")
    if cohort in (POST_COHORT, AI_COHORT) and observed_at < cutoff:
        raise ValueError(f"record {record.id!r} is not on or after the temporal cutoff")

    match_id_raw = raw.get("temporal_match_id")
    match_id = str(match_id_raw).strip() if match_id_raw else None
    if match_id and record.split != "test":
        raise ValueError(f"record {record.id!r} assigns a non-test group to a temporal match")
    matching_raw = raw.get("matching", {})
    if not isinstance(matching_raw, Mapping):
        raise ValueError(f"record {record.id!r} matching must be an object")
    matching = {str(key): str(value) for key, value in matching_raw.items()}
    return TemporalGroup(
        group_id=record.group_id,
        leakage_group_id=record.leakage_group_id,
        source=record.source,
        split=record.split,
        page_type=record.page_type,
        cohort=cohort,
        timestamp=observed_at,
        match_id=match_id,
        matching=matching,
    )


def _validate_panel(panel: Mapping[str, Any], protocol: Mapping[str, Any]) -> ValidatedPanel:
    validate_scale_temporal_protocol(protocol)
    if panel.get("schema") != PANEL_SCHEMA:
        raise ValueError(f"panel schema must be {PANEL_SCHEMA!r}")
    metadata = panel.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("panel requires metadata")
    if metadata.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("panel does not bind the supplied protocol")
    cutoff = _timestamp(protocol["temporal_cutoff"], "protocol.temporal_cutoff")
    if metadata.get("temporal_cutoff") != protocol["temporal_cutoff"]:
        raise ValueError("panel temporal cutoff differs from the protocol")
    capture_sha256 = _sha256(
        metadata.get("capture_protocol_sha256"), "metadata.capture_protocol_sha256"
    )
    _sha256(metadata.get("matching_receipt_sha256"), "metadata.matching_receipt_sha256")
    if metadata.get("matching_frozen_before_scoring") is not True:
        raise ValueError("panel matching must be frozen before scoring")
    if metadata.get("required_page_types") != protocol.get("required_page_types"):
        raise ValueError("panel required_page_types differ from the protocol")

    raw_records = panel.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("panel requires records")
    records = [record_from_dict(raw) for raw in raw_records]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("panel record ids must be unique")

    groups: dict[str, TemporalGroup] = {}
    leakage: dict[str, tuple[str, str]] = {}
    for raw, record in zip(raw_records, records, strict=True):
        observed = _group_metadata(raw, record, cutoff, capture_sha256)
        previous = groups.setdefault(record.group_id, observed)
        if previous != observed:
            raise ValueError(f"group {record.group_id!r} has inconsistent temporal metadata")
        previous_leakage = leakage.setdefault(
            record.leakage_group_id, (record.split, observed.cohort)
        )
        if previous_leakage != (record.split, observed.cohort):
            raise ValueError(
                f"leakage group {record.leakage_group_id!r} crosses a split or temporal cohort"
            )
    exact_fields = list(protocol["matching"]["exact_fields"])
    blocks: dict[str, dict[str, str]] = defaultdict(dict)
    for group in groups.values():
        if not group.match_id:
            continue
        if group.cohort in blocks[group.match_id]:
            raise ValueError(
                f"temporal match {group.match_id!r} contains two {group.cohort!r} groups"
            )
        blocks[group.match_id][group.cohort] = group.group_id
    for match_id, members in blocks.items():
        if set(members) != set(COHORTS):
            raise ValueError(f"temporal match {match_id!r} is incomplete")
        matched_groups = [groups[group_id] for group_id in members.values()]
        if len({group.page_type for group in matched_groups}) != 1:
            raise ValueError(f"temporal match {match_id!r} crosses page types")
        for field in exact_fields:
            values = {group.matching.get(field) for group in matched_groups}
            if None in values or len(values) != 1:
                raise ValueError(
                    f"temporal match {match_id!r} does not exactly match field {field!r}"
                )

    counts: dict[str, dict[str, dict[str, int]]] = {}
    match_counts = {page_type: 0 for page_type in protocol["required_page_types"]}
    for page_type in protocol["required_page_types"]:
        counts[page_type] = {}
        for cohort in COHORTS:
            counts[page_type][cohort] = {}
            for split in ("train", "dev", "test"):
                counts[page_type][cohort][split] = sum(
                    1
                    for group in groups.values()
                    if group.page_type == page_type
                    and group.cohort == cohort
                    and group.split == split
                )
    for members in blocks.values():
        page_type = groups[members[PRE_COHORT]].page_type
        if page_type in match_counts:
            match_counts[page_type] += 1

    deficits: list[dict[str, Any]] = []
    targets = protocol["targets_per_page_type"]
    for page_type in protocol["required_page_types"]:
        for cohort, split_targets in targets.items():
            for split, required in split_targets.items():
                observed = counts[page_type][cohort][split]
                if observed < required:
                    deficits.append(
                        {
                            "page_type": page_type,
                            "cohort": cohort,
                            "split": split,
                            "observed": observed,
                            "required": required,
                            "deficit": required - observed,
                        }
                    )
        required_matches = protocol["complete_temporal_match_blocks_per_page_type"]
        if match_counts[page_type] < required_matches:
            deficits.append(
                {
                    "page_type": page_type,
                    "cohort": "complete_temporal_match_block",
                    "split": "test",
                    "observed": match_counts[page_type],
                    "required": required_matches,
                    "deficit": required_matches - match_counts[page_type],
                }
            )

    audit = {
        "schema": "webmark_scale_temporal_panel_audit_v1",
        "status": "ready" if not deficits else "incomplete",
        "protocol_sha256": canonical_json_sha256(protocol),
        "panel_sha256": canonical_json_sha256(panel),
        "n_records": len(records),
        "n_groups": len(groups),
        "counts_by_page_type": counts,
        "complete_match_blocks_by_page_type": match_counts,
        "deficits": deficits,
        "counting_policy": (
            "Only parsed, capture-protocol-bound source or generation groups count; repeated captures "
            "remain one group and incomplete temporal blocks do not count."
        ),
    }
    return ValidatedPanel(tuple(records), groups, blocks, audit)


def audit_scale_temporal_panel(
    panel: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the fail-closed scale and temporal-panel coverage audit."""
    return dict(_validate_panel(panel, protocol).audit)


def build_scale_temporal_panel(
    benchmark_manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    capture_protocol_sha256: str,
) -> dict[str, Any]:
    """Join a benchmark manifest to a frozen temporal-cohort registry."""
    validate_scale_temporal_protocol(protocol)
    capture_protocol_sha256 = _sha256(capture_protocol_sha256, "capture_protocol_sha256")
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"registry schema must be {REGISTRY_SCHEMA!r}")
    registry_metadata = registry.get("metadata")
    if not isinstance(registry_metadata, Mapping):
        raise ValueError("registry requires metadata")
    if registry_metadata.get("matching_frozen_before_scoring") is not True:
        raise ValueError("registry matching must be frozen before scoring")
    raw_groups = registry.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("registry requires groups")
    registry_by_group: dict[str, Mapping[str, Any]] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping) or not raw_group.get("group_id"):
            raise ValueError("registry group requires group_id")
        group_id = str(raw_group["group_id"])
        if group_id in registry_by_group:
            raise ValueError(f"duplicate registry group {group_id!r}")
        registry_by_group[group_id] = raw_group

    raw_records = benchmark_manifest.get("records")
    metadata = benchmark_manifest.get("metadata")
    if not isinstance(raw_records, list) or not raw_records or not isinstance(metadata, Mapping):
        raise ValueError("benchmark manifest requires metadata and records")
    benchmark_groups = {str(record.get("group_id") or "") for record in raw_records}
    if "" in benchmark_groups or benchmark_groups != set(registry_by_group):
        missing = sorted(benchmark_groups.difference(registry_by_group))
        extra = sorted(set(registry_by_group).difference(benchmark_groups))
        raise ValueError(
            f"benchmark/registry group mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    records: list[dict[str, Any]] = []
    for raw_record in raw_records:
        group = registry_by_group[str(raw_record["group_id"])]
        record = dict(raw_record)
        record["temporal_cohort"] = group.get("temporal_cohort")
        if group.get("temporal_match_id"):
            record["temporal_match_id"] = group["temporal_match_id"]
        record["matching"] = dict(group.get("matching", {}))
        provenance = dict(record.get("provenance", {}))
        provenance["capture_protocol_sha256"] = capture_protocol_sha256
        cohort = group.get("temporal_cohort")
        timestamp_field = "generated_at" if cohort == AI_COHORT else "revision_timestamp"
        provenance[timestamp_field] = group.get(timestamp_field)
        record["provenance"] = provenance
        records.append(record)

    panel_metadata = dict(metadata)
    panel_metadata.update(
        {
            "protocol_sha256": canonical_json_sha256(protocol),
            "base_benchmark_manifest_sha256": canonical_json_sha256(benchmark_manifest),
            "temporal_cutoff": protocol["temporal_cutoff"],
            "capture_protocol_sha256": capture_protocol_sha256,
            "matching_receipt_sha256": canonical_json_sha256(registry),
            "matching_frozen_before_scoring": True,
            "required_page_types": list(protocol["required_page_types"]),
        }
    )
    return {"schema": PANEL_SCHEMA, "metadata": panel_metadata, "records": records}


def _score_groups(
    train_records: Sequence[BenchmarkRecord],
    evaluation_records: Sequence[BenchmarkRecord],
) -> dict[str, float]:
    references = fit_reference_by_page_type(train_records)
    scorers = {
        page_type: BiasScorer(reference, penalty="l2", nonparametric_dims=("typography",))
        for page_type, reference in references.items()
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in evaluation_records:
        grouped[record.group_id].append(scorers[record.page_type].score(record.features).total)
    return {group_id: statistics.fmean(values) for group_id, values in sorted(grouped.items())}


def _reference_auc(near_distances: Sequence[float], far_distances: Sequence[float]) -> float:
    if not near_distances or not far_distances:
        raise ValueError("reference-fit AUROC requires both cohorts")
    wins = 0.0
    for near in near_distances:
        for far in far_distances:
            wins += 1.0 if near < far else 0.5 if near == far else 0.0
    return wins / (len(near_distances) * len(far_distances))


def _matched_metrics(
    scores: Mapping[str, float],
    groups: Mapping[str, TemporalGroup],
    blocks: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    for match_id, members in sorted(blocks.items()):
        page_type = groups[members[PRE_COHORT]].page_type
        by_type[page_type].append(
            {
                "match_id": match_id,
                PRE_COHORT: scores[members[PRE_COHORT]],
                POST_COHORT: scores[members[POST_COHORT]],
                AI_COHORT: scores[members[AI_COHORT]],
            }
        )

    per_type: dict[str, dict[str, float | int]] = {}
    for page_type, rows in sorted(by_type.items()):
        pre = [float(row[PRE_COHORT]) for row in rows]
        post = [float(row[POST_COHORT]) for row in rows]
        ai = [float(row[AI_COHORT]) for row in rows]
        per_type[page_type] = {
            "n_match_blocks": len(rows),
            "pre_vs_ai_auc": _reference_auc(pre, ai),
            "pre_vs_post_auc": _reference_auc(pre, post),
            "post_vs_ai_auc": _reference_auc(post, ai),
            "mean_post_minus_pre_distance": statistics.fmean(
                post_value - pre_value for pre_value, post_value in zip(pre, post, strict=True)
            ),
            "mean_ai_minus_pre_distance": statistics.fmean(
                ai_value - pre_value for pre_value, ai_value in zip(pre, ai, strict=True)
            ),
            "mean_ai_minus_post_distance": statistics.fmean(
                ai_value - post_value for post_value, ai_value in zip(post, ai, strict=True)
            ),
            "ai_further_than_post_rate": statistics.fmean(
                1.0 if ai_value > post_value else 0.5 if ai_value == post_value else 0.0
                for post_value, ai_value in zip(post, ai, strict=True)
            ),
        }
    macro_fields = (
        "pre_vs_ai_auc",
        "pre_vs_post_auc",
        "post_vs_ai_auc",
        "mean_post_minus_pre_distance",
        "mean_ai_minus_pre_distance",
        "mean_ai_minus_post_distance",
        "ai_further_than_post_rate",
    )
    macro = {
        field: statistics.fmean(float(values[field]) for values in per_type.values())
        for field in macro_fields
    }
    return {"per_page_type": per_type, "macro": macro}


def _bootstrap_temporal(
    scores: Mapping[str, float],
    groups: Mapping[str, TemporalGroup],
    blocks: Mapping[str, Mapping[str, str]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    blocks_by_type: dict[str, list[tuple[str, Mapping[str, str]]]] = defaultdict(list)
    for match_id, members in blocks.items():
        blocks_by_type[groups[members[PRE_COHORT]].page_type].append((match_id, members))
    rng = random.Random(seed)
    sampled_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_resamples):
        sampled: dict[str, Mapping[str, str]] = {}
        for page_type, page_blocks in sorted(blocks_by_type.items()):
            for index in range(len(page_blocks)):
                match_id, members = rng.choice(page_blocks)
                sampled[f"{page_type}:{index}:{match_id}"] = members
        metrics = _matched_metrics(scores, groups, sampled)["macro"]
        for field, value in metrics.items():
            sampled_values[field].append(float(value))
    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "temporal_match_block_stratified_by_page_type",
        "macro": {field: _interval(values) for field, values in sampled_values.items()},
    }


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman correlation requires equal vectors with at least two values")
    rank_first = _ranks(first)
    rank_second = _ranks(second)
    mean_first = statistics.fmean(rank_first)
    mean_second = statistics.fmean(rank_second)
    numerator = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(rank_first, rank_second, strict=True)
    )
    denominator = math.sqrt(
        sum((value - mean_first) ** 2 for value in rank_first)
        * sum((value - mean_second) ** 2 for value in rank_second)
    )
    return numerator / denominator if denominator else 1.0


def _learning_curve(
    validated: ValidatedPanel,
    protocol: Mapping[str, Any],
    evaluation_records: Sequence[BenchmarkRecord],
    full_scores: Mapping[str, float],
    full_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    learning = protocol["learning_curve"]
    train_by_type: dict[str, dict[str, list[BenchmarkRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in validated.records:
        group = validated.groups[record.group_id]
        if group.cohort == PRE_COHORT and record.split == "train":
            train_by_type[record.page_type][record.group_id].append(record)
    evaluation_ids = sorted(full_scores)
    full_vector = [full_scores[group_id] for group_id in evaluation_ids]
    full_auc = float(full_metrics["macro"]["post_vs_ai_auc"])
    rng = random.Random(int(learning["seed"]))
    curve: list[dict[str, Any]] = []
    for train_n in learning["train_groups_per_type"]:
        available = {page_type: len(groups) for page_type, groups in train_by_type.items()}
        if any(
            available.get(page_type, 0) < train_n for page_type in protocol["required_page_types"]
        ):
            curve.append(
                {
                    "train_groups_per_type": train_n,
                    "status": "not_run",
                    "available_by_page_type": available,
                }
            )
            continue
        correlations: list[float] = []
        aucs: list[float] = []
        auc_drifts: list[float] = []
        excesses: list[float] = []
        for _ in range(int(learning["n_subsamples"])):
            selected_records: list[BenchmarkRecord] = []
            for page_type in protocol["required_page_types"]:
                group_ids = sorted(train_by_type[page_type])
                selected = rng.sample(group_ids, train_n)
                for group_id in selected:
                    selected_records.extend(train_by_type[page_type][group_id])
            scores = _score_groups(selected_records, evaluation_records)
            metrics = _matched_metrics(scores, validated.groups, validated.match_blocks)["macro"]
            vector = [scores[group_id] for group_id in evaluation_ids]
            auc = float(metrics["post_vs_ai_auc"])
            correlations.append(_spearman(vector, full_vector))
            aucs.append(auc)
            auc_drifts.append(abs(auc - full_auc))
            excesses.append(float(metrics["mean_ai_minus_post_distance"]))
        curve.append(
            {
                "train_groups_per_type": train_n,
                "status": "completed",
                "n_subsamples": int(learning["n_subsamples"]),
                "spearman_vs_full": _interval(correlations),
                "macro_post_vs_ai_auc": _interval(aucs),
                "macro_auc_absolute_drift_from_full": _interval(auc_drifts),
                "macro_mean_ai_minus_post_distance": _interval(excesses),
            }
        )

    stability_n = int(learning["stability_evaluation_train_groups_per_type"])
    stability_row = next(
        (row for row in curve if row["train_groups_per_type"] == stability_n), None
    )
    if not stability_row or stability_row["status"] != "completed":
        stability_status = "not_reportable"
    else:
        stable = stability_row["spearman_vs_full"]["ci_95"][0] >= float(
            learning["min_spearman_vs_full"]
        ) and stability_row["macro_auc_absolute_drift_from_full"]["ci_95"][1] <= float(
            learning["max_macro_auc_absolute_drift"]
        )
        stability_status = (
            "stable_at_preregistered_scale" if stable else "not_stable_at_preregistered_scale"
        )
    return {
        "status": stability_status,
        "full_reference_macro_post_vs_ai_auc": full_auc,
        "decision_thresholds": {
            "evaluation_train_groups_per_type": stability_n,
            "min_spearman_vs_full": learning["min_spearman_vs_full"],
            "max_macro_auc_absolute_drift": learning["max_macro_auc_absolute_drift"],
        },
        "curve": curve,
    }


def run_scale_temporal_validation(
    panel: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the preregistered scale and temporal-control experiment."""
    validated = _validate_panel(panel, protocol)
    if validated.audit["status"] != "ready":
        return {
            "schema": RESULT_SCHEMA,
            "status": "not_run",
            "protocol_sha256": canonical_json_sha256(protocol),
            "panel_audit": validated.audit,
            "reason": "The frozen scale and temporal-control coverage gate is incomplete.",
        }

    matched_group_ids = {
        group_id for members in validated.match_blocks.values() for group_id in members.values()
    }
    evaluation_records = [
        record for record in validated.records if record.group_id in matched_group_ids
    ]
    train_records = [
        record
        for record in validated.records
        if validated.groups[record.group_id].cohort == PRE_COHORT and record.split == "train"
    ]
    full_scores = _score_groups(train_records, evaluation_records)
    point = _matched_metrics(full_scores, validated.groups, validated.match_blocks)
    temporal_protocol = protocol["temporal_validation"]
    bootstrap = _bootstrap_temporal(
        full_scores,
        validated.groups,
        validated.match_blocks,
        n_resamples=int(temporal_protocol["n_bootstrap"]),
        seed=int(temporal_protocol["seed"]),
    )
    auc_ci_low = bootstrap["macro"]["post_vs_ai_auc"]["ci_95"][0]
    excess_ci_low = bootstrap["macro"]["mean_ai_minus_post_distance"]["ci_95"][0]
    all_types_positive = all(
        float(values["mean_ai_minus_post_distance"]) > 0
        for values in point["per_page_type"].values()
    )
    exceeds_temporal_drift = (
        auc_ci_low > float(temporal_protocol["min_post_vs_ai_auc_ci_low"])
        and excess_ci_low > float(temporal_protocol["min_ai_minus_post_distance_ci_low"])
        and (
            all_types_positive
            or not temporal_protocol.get("require_positive_effect_in_every_page_type")
        )
    )
    temporal_status = (
        "evidence_exceeds_observed_temporal_drift"
        if exceeds_temporal_drift
        else "temporal_confound_not_ruled_out"
    )
    learning_curve = _learning_curve(
        validated,
        protocol,
        evaluation_records,
        full_scores,
        point,
    )
    group_scores = [
        {
            "group_id": group_id,
            "page_type": validated.groups[group_id].page_type,
            "temporal_cohort": validated.groups[group_id].cohort,
            "temporal_match_id": validated.groups[group_id].match_id,
            "reference_distance": distance,
        }
        for group_id, distance in sorted(full_scores.items())
    ]
    return {
        "schema": RESULT_SCHEMA,
        "status": "completed",
        "protocol_sha256": canonical_json_sha256(protocol),
        "panel_sha256": canonical_json_sha256(panel),
        "panel_audit": validated.audit,
        "fit_cohort": PRE_COHORT,
        "fit_split": "train",
        "evaluation_unit": "source_or_generation_group",
        "temporal_validation": {
            "status": temporal_status,
            "point_estimates": point,
            "bootstrap": bootstrap,
            "all_page_types_positive_ai_minus_post_distance": all_types_positive,
            "decision_thresholds": {
                "min_post_vs_ai_auc_ci_low": temporal_protocol["min_post_vs_ai_auc_ci_low"],
                "min_ai_minus_post_distance_ci_low": temporal_protocol[
                    "min_ai_minus_post_distance_ci_low"
                ],
                "require_positive_effect_in_every_page_type": temporal_protocol.get(
                    "require_positive_effect_in_every_page_type", False
                ),
            },
        },
        "learning_curve": learning_curve,
        "group_scores": group_scores,
        "interpretation": (
            "A positive result shows that reference-distance separation exceeds the temporal drift "
            "observed in this matched open-source panel. It is not evidence of individual authorship, "
            "universal design quality, accessibility, or human preference."
        ),
    }
