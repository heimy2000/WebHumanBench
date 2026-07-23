"""Reference-relative design human-likeness benchmark utilities.

The benchmark measures proximity to a declared corpus of human-authored
webpages. It does not infer a page's true author and does not treat a small
panel's aesthetic preference as ground truth. ``train`` fits the reference,
``dev`` calibrates an empirical human-fit percentile within page type, and
``test`` is the only split used for human-versus-AI separation and model
reporting. Page type conditions the declared reference distribution; it is not
a learned controller input.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .bias import BiasScorer, ReferenceStats
from .features import FEATURE_NAMES, PageFeatures
from .page_type import PAGE_TYPES

SCHEMA = "webmark_human_likeness_benchmark_v1"
SPLITS = ("train", "dev", "test")
SOURCES = ("human", "ai")
MOBILE_MAX_WIDTH = 480
ZERO_VARIANCE_EPSILON = 1e-9


@dataclass(frozen=True)
class BenchmarkRecord:
    """One provenance-labelled webpage and its extracted CSS features."""

    id: str
    source: str
    split: str
    group_id: str
    leakage_group_id: str
    page_type: str
    viewport: str
    features: PageFeatures
    model_id: str | None = None


def _as_float_list(value: Any, field: str, record_id: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"record {record_id!r} requires a non-empty {field!r} list")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {record_id!r} has non-numeric {field!r} values") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"record {record_id!r} has non-finite {field!r} values")
    return values


def viewport_width(value: str) -> int:
    """Parse a ``WIDTHxHEIGHT`` viewport and return its positive width."""
    try:
        width, height = value.lower().split("x", 1)
        parsed_width, parsed_height = int(width), int(height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"viewport must use WIDTHxHEIGHT, got {value!r}") from exc
    if parsed_width <= 0 or parsed_height <= 0:
        raise ValueError(f"viewport dimensions must be positive, got {value!r}")
    return parsed_width


def record_from_dict(raw: Mapping[str, Any]) -> BenchmarkRecord:
    """Parse one JSON record and reject incomplete feature rows."""
    required = ("id", "source", "split", "group_id", "page_type", "viewport", "features")
    missing = [field for field in required if not raw.get(field)]
    if missing:
        raise ValueError(f"benchmark record is missing required fields: {', '.join(missing)}")

    record_id = str(raw["id"])
    source = str(raw["source"])
    split = str(raw["split"])
    page_type = str(raw["page_type"])
    if source not in SOURCES:
        raise ValueError(f"record {record_id!r} has unsupported source {source!r}")
    if split not in SPLITS:
        raise ValueError(f"record {record_id!r} has unsupported split {split!r}")
    if page_type not in PAGE_TYPES:
        raise ValueError(f"record {record_id!r} has unsupported page_type {page_type!r}")
    viewport = str(raw["viewport"])
    viewport_width(viewport)
    if source == "ai" and not raw.get("model_id"):
        raise ValueError(f"AI record {record_id!r} requires model_id")

    features = raw["features"]
    if not isinstance(features, Mapping):
        raise ValueError(f"record {record_id!r} has invalid features")
    color = features.get("color")
    if not isinstance(color, list) or not color or not all(isinstance(value, str) for value in color):
        raise ValueError(f"record {record_id!r} requires a non-empty string color list")

    return BenchmarkRecord(
        id=record_id,
        source=source,
        split=split,
        group_id=str(raw["group_id"]),
        leakage_group_id=str(raw.get("leakage_group_id", raw["group_id"])),
        page_type=page_type,
        viewport=viewport,
        model_id=str(raw["model_id"]) if raw.get("model_id") else None,
        features=PageFeatures(
            typography=_as_float_list(features.get("typography"), "typography", record_id),
            spacing=_as_float_list(features.get("spacing"), "spacing", record_id),
            grid=_as_float_list(features.get("grid"), "grid", record_id),
            color=[value.lower() for value in color],
            saturation=_as_float_list(features.get("saturation"), "saturation", record_id),
        ),
    )


def validate_manifest(manifest: Mapping[str, Any]) -> list[BenchmarkRecord]:
    """Validate the split, provenance, coverage, and leakage contract."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA!r}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("manifest requires a metadata object")
    for field in ("benchmark_name", "version", "reference_scope", "license"):
        if not metadata.get(field):
            raise ValueError(f"manifest metadata requires {field!r}")
    mobile_test_share_min = metadata.get("mobile_test_share_min")
    if not isinstance(mobile_test_share_min, int | float) or not 0 < mobile_test_share_min <= 1:
        raise ValueError("metadata requires mobile_test_share_min in (0, 1]")

    required_page_types = metadata.get("required_page_types")
    if not isinstance(required_page_types, list) or not required_page_types:
        raise ValueError("metadata requires a non-empty required_page_types list")
    unknown_types = set(required_page_types).difference(PAGE_TYPES)
    if unknown_types:
        raise ValueError(f"metadata has unsupported required_page_types: {sorted(unknown_types)}")

    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("manifest requires a non-empty records list")
    records = [record_from_dict(raw) for raw in raw_records]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("record ids must be unique")

    # ``group_id`` is the scoring unit. ``leakage_group_id`` is a broader
    # lineage boundary: it keeps, for example, one prompt family rendered by
    # several models from leaking across splits without mixing models in a
    # model-specific score summary.
    groups: dict[str, tuple[str, str, str | None, str]] = {}
    group_splits: dict[str, str] = {}
    leakage_splits: dict[str, str] = {}
    for record in records:
        group_fields = (record.source, record.page_type, record.model_id, record.leakage_group_id)
        previous = groups.setdefault(record.group_id, group_fields)
        if previous != group_fields:
            raise ValueError(
                f"group_id {record.group_id!r} mixes source, page_type, model_id, or leakage_group_id values; "
                "a reporting group must have one provenance and stratum"
            )
        previous_group_split = group_splits.setdefault(record.group_id, record.split)
        if previous_group_split != record.split:
            raise ValueError(
                f"group_id {record.group_id!r} occurs in both {previous_group_split!r} and {record.split!r}; "
                "a scoring group may not cross splits"
            )
        previous_split = leakage_splits.setdefault(record.leakage_group_id, record.split)
        if previous_split != record.split:
            raise ValueError(
                f"leakage_group_id {record.leakage_group_id!r} occurs in both "
                f"{previous_split!r} and {record.split!r}; site, template, or prompt leakage across splits is forbidden"
            )

    for split in ("train", "dev"):
        if any(record.source != "human" for record in records if record.split == split):
            raise ValueError(f"{split} may contain only human-authored reference pages")

    test_sources = {record.source for record in records if record.split == "test"}
    if test_sources != set(SOURCES):
        raise ValueError("test must contain both human and AI provenance-labelled pages")
    test = [record for record in records if record.split == "test"]
    mobile_test_share = sum(viewport_width(record.viewport) <= MOBILE_MAX_WIDTH for record in test) / len(test)
    if mobile_test_share < mobile_test_share_min:
        raise ValueError(
            f"test mobile share {mobile_test_share:.3f} is below required minimum {mobile_test_share_min:.3f}"
        )

    for split in SPLITS:
        observed = {record.page_type for record in records if record.split == split}
        missing_types = set(required_page_types).difference(observed)
        if missing_types:
            raise ValueError(f"{split} is missing required page types: {sorted(missing_types)}")
    for page_type in required_page_types:
        train_groups = {
            record.group_id
            for record in records
            if record.source == "human" and record.split == "train" and record.page_type == page_type
        }
        dev_groups = {
            record.group_id
            for record in records
            if record.source == "human" and record.split == "dev" and record.page_type == page_type
        }
        if len(train_groups) < 2:
            raise ValueError(
                f"train page type {page_type!r} requires at least two human source groups "
                "to fit a reference distribution"
            )
        if not dev_groups:
            raise ValueError(
                f"dev page type {page_type!r} requires at least one human source group "
                "for percentile calibration"
            )
    for page_type in required_page_types:
        test_sources_for_type = {
            record.source for record in records if record.split == "test" and record.page_type == page_type
        }
        if test_sources_for_type != set(SOURCES):
            raise ValueError(
                f"test page type {page_type!r} must contain both human and AI provenance records"
            )
    return records


def _page_summary(features: PageFeatures, dimension: str) -> float:
    if dimension == "color":
        return float(len({color.lower() for color in features.color if color}))
    values = getattr(features, dimension)
    return statistics.fmean(values)


def fit_reference(records: Iterable[BenchmarkRecord]) -> ReferenceStats:
    """Fit reference parameters from human train records only."""
    train = list(records)
    if len(train) < 2 or any(record.source != "human" or record.split != "train" for record in train):
        raise ValueError("reference fitting requires at least two human train records")
    groups: dict[str, list[BenchmarkRecord]] = {}
    for record in train:
        groups.setdefault(record.group_id, []).append(record)
    if len(groups) < 2:
        raise ValueError("reference fitting requires at least two human train groups")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for dimension in FEATURE_NAMES:
        values = [
            statistics.fmean(_page_summary(record.features, dimension) for record in members)
            for members in groups.values()
        ]
        std = statistics.stdev(values)
        means[dimension] = statistics.fmean(values)
        # A small pilot can have a constant train-only feature statistic. Keep
        # the exact zero so the scorer can exclude that unidentifiable
        # dimension instead of introducing an arbitrary scale.
        stds[dimension] = std if std > ZERO_VARIANCE_EPSILON else 0.0
    typography_samples: list[float] = []
    for members in groups.values():
        combined = sorted(sample for record in members for sample in record.features.typography)
        if not combined:
            continue
        # Five equally weighted quantiles per group prevent a page with more
        # captures or text nodes from dominating the reference distribution.
        for quantile in (0.05, 0.25, 0.5, 0.75, 0.95):
            index = min(len(combined) - 1, round(quantile * (len(combined) - 1)))
            typography_samples.append(combined[index])
    return ReferenceStats(
        means=means,
        stds=stds,
        wasserstein_samples_per_dim={"typography": typography_samples},
    )


def fit_reference_by_page_type(records: Iterable[BenchmarkRecord]) -> dict[str, ReferenceStats]:
    """Fit independent human references for each declared page type.

    This conditions the measurement target on a curator-assigned benchmark
    stratum without using page type to route or tune the CSS controller.
    """
    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.page_type, []).append(record)
    return {page_type: fit_reference(group) for page_type, group in sorted(grouped.items())}


def human_fit_percentile(score: float, calibration_scores: Sequence[float]) -> float:
    """Empirical percentile where larger means closer to the human reference."""
    if not calibration_scores:
        raise ValueError("at least one human dev score is required for calibration")
    greater = sum(value > score for value in calibration_scores)
    equal = sum(value == score for value in calibration_scores)
    return 100.0 * (greater + 0.5 * equal) / len(calibration_scores)


def roc_auc_human_vs_ai(
    rows: Sequence[Mapping[str, Any]], *, score_field: str = "human_fit_percentile"
) -> float:
    """Pairwise AUROC for a calibrated human-fit score.

    Human-fit percentiles are comparable across type-conditioned references;
    raw bias scores are not necessarily on a common scale across page types.
    """
    human = [float(row[score_field]) for row in rows if row["source"] == "human"]
    ai = [float(row[score_field]) for row in rows if row["source"] == "ai"]
    if not human or not ai:
        raise ValueError("AUROC requires both human and AI test rows")
    wins = 0.0
    for human_score in human:
        for ai_score in ai:
            wins += 1.0 if human_score > ai_score else 0.5 if human_score == ai_score else 0.0
    return wins / (len(human) * len(ai))


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    scores = [float(row["bias_score"]) for row in rows]
    percentiles = [float(row["human_fit_percentile"]) for row in rows]
    return {
        "n": len(rows),
        "mean_bias_score": statistics.fmean(scores),
        "median_bias_score": statistics.median(scores),
        "mean_human_fit_percentile": statistics.fmean(percentiles),
    }


def _group_score_rows(
    rows: Sequence[Mapping[str, Any]],
    calibration_scores_by_page_type: Mapping[str, Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate repeated captures before reporting the benchmark endpoint.

    Group-level aggregation prevents a source with several captures from
    silently receiving more weight than another source. The raw rows remain in
    the output for diagnostics, but all reported benchmark summaries use these
    group rows.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)

    output: list[dict[str, Any]] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        first = members[0]
        score = statistics.fmean(float(member["bias_score"]) for member in members)
        dimensions = {
            dimension: statistics.fmean(float(member["per_dimension"][dimension]) for member in members)
            for dimension in FEATURE_NAMES
        }
        item: dict[str, Any] = {
            "group_id": group_id,
            "leakage_group_id": first["leakage_group_id"],
            "source": first["source"],
            "page_type": first["page_type"],
            "model_id": first["model_id"],
            "n_records": len(members),
            "bias_score": score,
            "per_dimension": dimensions,
        }
        if calibration_scores_by_page_type is not None:
            page_type = str(first["page_type"])
            calibration_scores = calibration_scores_by_page_type.get(page_type)
            if calibration_scores is None:
                raise ValueError(f"missing development calibration scores for page type {page_type!r}")
            item["human_fit_percentile"] = human_fit_percentile(score, calibration_scores)
        output.append(item)
    return output


def _calibration_scope(
    dev_groups: Sequence[Mapping[str, Any]], test_groups: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Expose finite calibration resolution before interpreting an AUROC."""
    page_types = sorted({str(row["page_type"]) for row in test_groups})
    per_page_type: dict[str, dict[str, int]] = {}
    for page_type in page_types:
        n_dev = sum(row["page_type"] == page_type for row in dev_groups)
        n_human_test = sum(
            row["page_type"] == page_type and row["source"] == "human" for row in test_groups
        )
        per_page_type[page_type] = {
            "n_dev_human_groups": n_dev,
            "n_test_human_groups": n_human_test,
            "max_midrank_percentile_levels": 2 * n_dev + 1,
        }
    limited = [
        page_type
        for page_type, counts in per_page_type.items()
        if counts["n_dev_human_groups"] == 1 or counts["n_test_human_groups"] < 5
    ]
    note = (
        "AUROC is a corpus-conditional descriptive rank statistic, not a human-authorship detector or a "
        "cross-source performance estimate."
    )
    if limited:
        note += (
            " Small development or human-test group counts make percentile calibration and AUROC especially "
            f"unstable for: {', '.join(limited)}."
        )
    return {"per_page_type": per_page_type, "limited_page_types": limited, "note": note}


def evaluate_human_likeness_benchmark(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run the leakage-safe reference-relative benchmark evaluation."""
    records = validate_manifest(manifest)
    train = [record for record in records if record.split == "train"]
    dev = [record for record in records if record.split == "dev"]
    test = [record for record in records if record.split == "test"]
    references = fit_reference_by_page_type(train)
    scorers = {
        page_type: BiasScorer(reference, penalty="l2", nonparametric_dims=("typography",))
        for page_type, reference in references.items()
    }
    calibration_rows: list[dict[str, Any]] = []
    for record in dev:
        components = scorers[record.page_type].score(record.features)
        calibration_rows.append({
            "group_id": record.group_id,
            "leakage_group_id": record.leakage_group_id,
            "source": record.source,
            "page_type": record.page_type,
            "model_id": record.model_id,
            "bias_score": components.total,
            "per_dimension": components.per_dim,
        })
    # Development calibration follows the same group unit as the reported test
    # endpoint. This avoids a repeated viewport capture changing a percentile.
    dev_groups = _group_score_rows(calibration_rows)
    calibration_scores_by_page_type = {
        page_type: [float(row["bias_score"]) for row in dev_groups if row["page_type"] == page_type]
        for page_type in references
    }

    rows: list[dict[str, Any]] = []
    for record in test:
        components = scorers[record.page_type].score(record.features)
        rows.append({
            "id": record.id,
            "group_id": record.group_id,
            "leakage_group_id": record.leakage_group_id,
            "source": record.source,
            "page_type": record.page_type,
            "model_id": record.model_id,
            "viewport": record.viewport,
            "bias_score": components.total,
            "human_fit_percentile": human_fit_percentile(
                components.total, calibration_scores_by_page_type[record.page_type]
            ),
            "per_dimension": components.per_dim,
        })

    group_rows = _group_score_rows(rows, calibration_scores_by_page_type)
    calibration_scope = _calibration_scope(dev_groups, group_rows)
    by_source = {
        source: _summary([row for row in group_rows if row["source"] == source])
        for source in SOURCES
    }
    by_page_type = {}
    for page_type in sorted({row["page_type"] for row in group_rows}):
        page_rows = [row for row in group_rows if row["page_type"] == page_type]
        by_page_type[page_type] = {
            **_summary(page_rows),
            "human_vs_ai_auroc": roc_auc_human_vs_ai(page_rows),
        }
    by_model = {
        model: _summary([row for row in group_rows if row["model_id"] == model])
        for model in sorted({row["model_id"] for row in group_rows if row["model_id"]})
    }
    by_viewport = {
        viewport: _summary([row for row in rows if row["viewport"] == viewport])
        for viewport in sorted({row["viewport"] for row in rows})
    }
    return {
        "schema": SCHEMA,
        "benchmark": manifest["metadata"]["benchmark_name"],
        "version": manifest["metadata"]["version"],
        "reference": {
            "fit_split": "train",
            "calibration_split": "dev",
            "evaluation_split": "test",
            "mode": "page_type_conditioned",
            "by_page_type": {
                page_type: {
                    "means": reference.means,
                    "stds": reference.stds,
                    "inactive_zero_variance_dimensions": sorted(
                        dimension
                        for dimension, std in reference.stds.items()
                        if std <= ZERO_VARIANCE_EPSILON
                    ),
                    "n_train_human": sum(record.page_type == page_type for record in train),
                    "n_dev_human": sum(record.page_type == page_type for record in dev),
                    "n_train_groups": len({record.group_id for record in train if record.page_type == page_type}),
                    "n_dev_groups": len({record.group_id for record in dev if record.page_type == page_type}),
                }
                for page_type, reference in references.items()
            },
            "n_train_human": len(train),
            "n_dev_human": len(dev),
            "n_train_groups": len({record.group_id for record in train}),
            "n_dev_groups": len({record.group_id for record in dev}),
            "zero_variance_policy": (
                "A feature with zero variation across the train-only source groups in a page type is "
                "excluded from normalized distance scoring; this is an insufficient-variation flag, not "
                "an exact-match rule."
            ),
        },
        "test": {
            "scoring_unit": "group",
            "n_rows": len(rows),
            "n_groups": len(group_rows),
            "human_vs_ai_auroc": roc_auc_human_vs_ai(group_rows),
            "by_source": by_source,
            "by_page_type": by_page_type,
            "by_model": by_model,
            "by_viewport": by_viewport,
            "by_viewport_note": (
                "Viewport summaries use raw capture rows for diagnostics only; "
                "the benchmark endpoint is aggregated at the group level."
            ),
            "calibration_scope": calibration_scope,
            "groups": group_rows,
            "rows": rows,
        },
        "interpretation": (
            "Scores measure fit to the declared human reference distribution. "
            "They do not identify true authorship, establish aesthetic preference, "
            "or justify a normative claim about human design."
        ),
    }
