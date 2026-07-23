"""Candidate-cohort design profiles for WebHumanBench corpus construction.

The profiles in this module summarize browser-computed CSS measurements from
open-reference *candidates*. They support exploratory characterization of a
versioned candidate cohort, but do not establish human authorship, design
quality, or a completed WebHumanBench reference corpus.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .open_reference import (
    HISTORICAL_EVIDENCE_SCHEMA,
    validate_open_reference_manifest,
)
from .release import canonical_json_sha256
from .vendor_snapshot import SNAPSHOT_CAPTURE_METHOD

PROBE_SCHEMA = "webmark_open_reference_candidate_mobile_probe_v1"
PROFILE_SCHEMA = "webmark_candidate_design_profile_v1"
REFERENCE_PROFILE_SCHEMA = "webmark_reference_design_profile_v1"
PRIMARY_VIEWPORTS = ("390x844", "430x932")
NEUTRAL_SATURATION_MAX = 0.10
GRID_SNAP_TOLERANCE = 0.125

PROFILE_METRICS = (
    "font_size_p50_px",
    "font_size_iqr_px",
    "type_hierarchy_ratio",
    "line_height_p50",
    "line_height_iqr",
    "grid_8px_snap_rate",
    "palette_unique_count",
    "palette_top5_share",
    "neutral_color_share",
    "saturation_p50",
)


def _finite_numbers(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"features.{field} must be a non-empty list")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"features.{field} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"features.{field} contains a non-finite value")
    return values


def _string_values(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"features.{field} must be a non-empty string list")
    return [item.lower() for item in value]


def _quantile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated quantile."""
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _iqr(values: Sequence[float]) -> float:
    return _quantile(values, 0.75) - _quantile(values, 0.25)


def profile_features(features: Mapping[str, Any]) -> dict[str, float | int]:
    """Compute interpretable, viewport-conditioned CSS profile measurements.

    ``grid`` is the extractor's left-position modulo-12 phase. Its snap rate is
    therefore an 8px-phase proxy, not proof of an explicit CSS grid system.
    """
    typography = _finite_numbers(features.get("typography"), "typography")
    spacing = _finite_numbers(features.get("spacing"), "spacing")
    grid = _finite_numbers(features.get("grid"), "grid")
    colors = _string_values(features.get("color"), "color")
    saturation = _finite_numbers(features.get("saturation"), "saturation")

    font_p50 = _quantile(typography, 0.50)
    if font_p50 <= 0:
        raise ValueError("features.typography has a non-positive median")
    palette = Counter(colors)
    top5_share = sum(count for _, count in palette.most_common(5)) / len(colors)
    snapped = sum(abs(value - round(value)) <= GRID_SNAP_TOLERANCE for value in grid) / len(grid)

    return {
        "text_style_sample_count": len(typography),
        "font_size_p10_px": _quantile(typography, 0.10),
        "font_size_p50_px": font_p50,
        "font_size_p90_px": _quantile(typography, 0.90),
        "font_size_iqr_px": _iqr(typography),
        "type_hierarchy_ratio": _quantile(typography, 0.90) / font_p50,
        "line_height_sample_count": len(spacing),
        "line_height_p50": _quantile(spacing, 0.50),
        "line_height_iqr": _iqr(spacing),
        "grid_phase_sample_count": len(grid),
        "grid_8px_snap_rate": snapped,
        "palette_sample_count": len(colors),
        "palette_unique_count": len(palette),
        "palette_top5_share": top5_share,
        "saturation_sample_count": len(saturation),
        "neutral_color_share": sum(value <= NEUTRAL_SATURATION_MAX for value in saturation) / len(saturation),
        "saturation_p50": _quantile(saturation, 0.50),
        "saturation_iqr": _iqr(saturation),
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median": _quantile(values, 0.50),
        "iqr": _iqr(values),
    }


def summarize_profiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize source groups without pooling CSS samples or double-counting viewports."""
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        profile = row["profile"]
        if not isinstance(profile, Mapping):
            raise ValueError("profile rows require a profile object")
        by_source[str(row["source_group"])].append(profile)

    source_profiles: list[dict[str, float]] = []
    for profiles in by_source.values():
        source_profiles.append(
            {
                metric: sum(float(profile[metric]) for profile in profiles) / len(profiles)
                for metric in PROFILE_METRICS
            }
        )
    metric_summaries: dict[str, dict[str, float | int]] = {}
    for metric in PROFILE_METRICS:
        values = [profile[metric] for profile in source_profiles]
        metric_summaries[metric] = _summary(values)
    return {
        "source_groups": len(source_profiles),
        "captures": len(rows),
        "aggregation": "source_group_mean_over_retained_captures_v1",
        "metrics": metric_summaries,
    }


def _source_group(row: Mapping[str, Any]) -> str:
    required = ("task_id", "repository", "commit_sha")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"candidate probe is missing: {', '.join(missing)}")
    return "::".join(str(row[field]) for field in required)


def build_candidate_design_profile(probe_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Analyze retained computed features from an unverified candidate probe.

    Rows without a complete browser capture or retained raw features are
    excluded and counted. The output deliberately remains a candidate-cohort
    report, even when every source has a pinned commit and mobile screenshot.
    """
    if probe_payload.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"expected probe schema {PROBE_SCHEMA!r}")
    raw_probes = probe_payload.get("probes")
    if not isinstance(raw_probes, list):
        raise ValueError("candidate probe payload requires a probes list")

    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for raw in raw_probes:
        if not isinstance(raw, Mapping):
            exclusions["malformed_probe"] += 1
            continue
        if raw.get("probe_status") != "captured":
            exclusions[f"probe_status:{raw.get('probe_status', 'missing')}"] += 1
            continue
        if not isinstance(raw.get("features"), Mapping):
            exclusions["missing_retained_features"] += 1
            continue
        try:
            source_group = _source_group(raw)
            viewport = str(raw["viewport"])
            if viewport not in PRIMARY_VIEWPORTS:
                exclusions[f"unsupported_viewport:{viewport}"] += 1
                continue
            key = (source_group, viewport)
            if key in seen:
                exclusions["duplicate_source_viewport"] += 1
                continue
            seen.add(key)
            profile = profile_features(raw["features"])
            rows.append(
                {
                    "source_group": source_group,
                    "task_id": str(raw["task_id"]),
                    "repository": str(raw["repository"]),
                    "commit_sha": str(raw["commit_sha"]),
                    "page_type": str(raw["page_type"]),
                    "viewport": viewport,
                    "rendered_url": raw.get("rendered_url"),
                    "screenshot_sha256": raw.get("screenshot_sha256"),
                    "feature_sha256": raw.get("feature_sha256"),
                    "profile": profile,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            exclusions[f"invalid_features:{str(exc)}"] += 1

    if not rows:
        raise ValueError(
            "no analyzable candidate captures; rerun the mobile probe with --retain-features"
        )

    by_viewport = {
        viewport: summarize_profiles([row for row in rows if row["viewport"] == viewport])
        for viewport in PRIMARY_VIEWPORTS
        if any(row["viewport"] == viewport for row in rows)
    }
    by_page_type: dict[str, dict[str, Any]] = {}
    for page_type in sorted({str(row["page_type"]) for row in rows}):
        page_rows = [row for row in rows if row["page_type"] == page_type]
        by_page_type[page_type] = summarize_profiles(page_rows)

    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[row["source_group"]][row["viewport"]] = row
    paired_deltas: dict[str, list[float]] = {metric: [] for metric in PROFILE_METRICS}
    for captures in paired.values():
        if not all(viewport in captures for viewport in PRIMARY_VIEWPORTS):
            continue
        smaller = captures[PRIMARY_VIEWPORTS[0]]["profile"]
        larger = captures[PRIMARY_VIEWPORTS[1]]["profile"]
        for metric in PROFILE_METRICS:
            paired_deltas[metric].append(abs(float(larger[metric]) - float(smaller[metric])))

    paired_summary = {
        "paired_source_groups": len(next(iter(paired_deltas.values()), [])),
        "absolute_delta": {
            metric: _summary(values) for metric, values in paired_deltas.items() if values
        },
    }
    return {
        "schema": PROFILE_SCHEMA,
        "data_status": "unverified_current_remote_candidate_cohort",
        "scope": (
            "Viewport-conditioned computed CSS profiles from current remote candidate deployments. "
            "They do not prove human authorship, deployment-to-commit or historical-time parity, "
            "preference, or a completed WebHumanBench reference corpus."
        ),
        "input_probe_schema": PROBE_SCHEMA,
        "source_groups": len({row["source_group"] for row in rows}),
        "captures": len(rows),
        "exclusions": dict(sorted(exclusions.items())),
        "overall": summarize_profiles(rows),
        "by_viewport": by_viewport,
        "by_page_type": by_page_type,
        "paired_mobile_viewport_deltas": paired_summary,
        "records": rows,
    }


def _metric_cell(summary: Mapping[str, Any], metric: str) -> str:
    row = summary.get("metrics", {}).get(metric)
    if not isinstance(row, Mapping):
        return "n/a"
    return f"{float(row['median']):.3f} (IQR {float(row['iqr']):.3f})"


def candidate_design_profile_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise, scope-safe Markdown companion for a profile report."""
    if report.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"expected profile schema {PROFILE_SCHEMA!r}")
    overall = report["overall"]
    lines = [
        "# Candidate Cohort Design Profile",
        "",
        "This report summarizes browser-computed CSS measurements from an unverified current remote candidate cohort.",
        "It is not evidence of human authorship, historical commit parity, preference, universal design quality, or a completed WebHumanBench corpus.",
        "",
        f"- Source groups: {report['source_groups']}",
        f"- Mobile captures: {report['captures']}",
        f"- Paired 390x844 / 430x932 source groups: {report['paired_mobile_viewport_deltas']['paired_source_groups']}",
        "- Overall medians first average each source group across its retained mobile captures.",
        "",
        "## Overall Mobile Profile",
        "",
        "| Metric | Source-level median (IQR) |",
        "| --- | --- |",
    ]
    labels = {
        "font_size_p50_px": "Text font-size p50 (px)",
        "font_size_iqr_px": "Text font-size IQR (px)",
        "type_hierarchy_ratio": "Type hierarchy p90 / p50",
        "line_height_p50": "Line-height ratio p50",
        "line_height_iqr": "Line-height ratio IQR",
        "grid_8px_snap_rate": "8px phase-alignment proxy",
        "palette_unique_count": "Unique computed colors",
        "palette_top5_share": "Top-5 color share",
        "neutral_color_share": "Neutral color share (S <= 0.10)",
        "saturation_p50": "Saturation p50",
    }
    for metric in PROFILE_METRICS:
        lines.append(f"| {labels[metric]} | {_metric_cell(overall, metric)} |")

    lines.extend(["", "## By Page Type", "", "| Page type | Sources | Captures | Type hierarchy | Neutral share |", "| --- | ---: | ---: | --- | --- |"])
    for page_type, summary in sorted(report["by_page_type"].items()):
        lines.append(
            "| "
            f"{page_type} | {summary['source_groups']} | {summary['captures']} | "
            f"{_metric_cell(summary, 'type_hierarchy_ratio')} | "
            f"{_metric_cell(summary, 'neutral_color_share')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The measures describe the captured candidate cohort only. A remote deployment may postdate or differ from its pinned source commit. The 8px value is a left-position phase proxy; palette and saturation values are computed-style samples; and dual-viewport differences describe responsive change rather than a quality score. A source enters the benchmark only after source-level provenance review, entrypoint/build verification, fixed-commit capture evidence, and the independent-review requirements of the public-release audit.",
            "",
        ]
    )
    return "\n".join(lines)


def _profile_rollups(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_viewport = {
        viewport: summarize_profiles([row for row in rows if row["viewport"] == viewport])
        for viewport in PRIMARY_VIEWPORTS
        if any(row["viewport"] == viewport for row in rows)
    }
    by_page_type: dict[str, dict[str, Any]] = {}
    for page_type in sorted({str(row["page_type"]) for row in rows}):
        by_page_type[page_type] = summarize_profiles([row for row in rows if row["page_type"] == page_type])

    paired: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[str(row["source_group"])][str(row["viewport"])] = row
    paired_deltas: dict[str, list[float]] = {metric: [] for metric in PROFILE_METRICS}
    for captures in paired.values():
        if not all(viewport in captures for viewport in PRIMARY_VIEWPORTS):
            continue
        smaller = captures[PRIMARY_VIEWPORTS[0]]["profile"]
        larger = captures[PRIMARY_VIEWPORTS[1]]["profile"]
        for metric in PROFILE_METRICS:
            paired_deltas[metric].append(abs(float(larger[metric]) - float(smaller[metric])))
    paired_summary = {
        "paired_source_groups": len(next(iter(paired_deltas.values()), [])),
        "absolute_delta": {
            metric: _summary(values) for metric, values in paired_deltas.items() if values
        },
    }
    return by_viewport, by_page_type, paired_summary


def build_reference_design_profile(
    source_manifest: Mapping[str, Any], capture_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize a complete fixed-commit reference capture without a preference claim.

    This is deliberately narrower than a release audit: it requires the
    manifest/capture identity, both declared mobile viewports, and local-only
    source captures, then derives descriptive source-level CSS statistics. The
    caller should run the full public-release audit separately before labelling
    the resulting cohort as a published benchmark reference.
    """
    if source_manifest.get("schema") != HISTORICAL_EVIDENCE_SCHEMA:
        raise ValueError("reference design profile requires a historical-evidence source manifest")
    sources = validate_open_reference_manifest(source_manifest)
    metadata = source_manifest.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("release_status") != "public":
        raise ValueError("reference design profile requires a public source manifest")
    if capture_ledger.get("schema") != "webmark_open_mobile_capture_v2":
        raise ValueError("reference design profile requires a webmark_open_mobile_capture_v2 ledger")
    if capture_ledger.get("source_manifest_sha256") != canonical_json_sha256(source_manifest):
        raise ValueError("capture ledger source_manifest_sha256 does not match the source manifest")
    if capture_ledger.get("status") not in {None, "complete"}:
        raise ValueError("reference design profile requires a complete capture ledger")
    if capture_ledger.get("failures"):
        raise ValueError("reference design profile cannot use a capture ledger with failures")
    raw_records = capture_ledger.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("capture ledger requires a records list")

    source_by_id = {source.id: source for source in sources}
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("capture ledger record must be an object")
        source_id = str(raw_record.get("source_id", ""))
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"capture ledger references unknown source_id {source_id!r}")
        viewport = str(raw_record.get("viewport", ""))
        if viewport not in PRIMARY_VIEWPORTS:
            continue
        key = (source_id, viewport)
        if key in seen:
            raise ValueError(f"capture ledger has duplicate source/viewport {key!r}")
        seen.add(key)
        if raw_record.get("source") != "human" or raw_record.get("group_id") != source.group_id:
            raise ValueError(f"capture {source_id!r}@{viewport} does not match its source identity")
        if raw_record.get("page_type") != source.page_type:
            raise ValueError(f"capture {source_id!r}@{viewport} does not match its page type")
        origin = raw_record.get("capture_origin")
        if not isinstance(origin, Mapping) or origin.get("mode") not in {
            "pinned_local_static_checkout",
            SNAPSHOT_CAPTURE_METHOD,
            "pinned_local_build",
        }:
            raise ValueError(f"capture {source_id!r}@{viewport} is not a fixed local source capture")
        if raw_record.get("blocked_external_requests"):
            raise ValueError(f"capture {source_id!r}@{viewport} retained blocked external requests")
        features = raw_record.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"capture {source_id!r}@{viewport} lacks feature arrays")
        rows.append({
            "source_group": source.group_id,
            "source_id": source_id,
            "page_type": source.page_type,
            "viewport": viewport,
            "capture_id": raw_record.get("id"),
            "feature_sha256": raw_record.get("feature_sha256"),
            "capture_html_sha256": raw_record.get("capture_html_sha256"),
            "profile": profile_features(features),
        })
    for source in sources:
        for viewport in PRIMARY_VIEWPORTS:
            if (source.id, viewport) not in seen:
                raise ValueError(f"source {source.id!r} lacks reference capture {viewport!r}")
    if not rows:
        raise ValueError("reference design profile has no primary mobile captures")

    by_viewport, by_page_type, paired_summary = _profile_rollups(rows)
    return {
        "schema": REFERENCE_PROFILE_SCHEMA,
        "data_status": "fixed_commit_historical_open_source_reference_cohort",
        "scope": (
            "Viewport-conditioned computed CSS profiles from the declared fixed-commit historical "
            "open-source reference cohort. The historical evidence policy is a temporal/source-provenance proxy, "
            "not proof of individual human authorship, preference, or universal design quality."
        ),
        "source_manifest_sha256": canonical_json_sha256(source_manifest),
        "capture_ledger_sha256": canonical_json_sha256(capture_ledger),
        "feature_extractor_version": capture_ledger.get("feature_extractor_version"),
        "source_groups": len({row["source_group"] for row in rows}),
        "captures": len(rows),
        "overall": summarize_profiles(rows),
        "by_viewport": by_viewport,
        "by_page_type": by_page_type,
        "paired_mobile_viewport_deltas": paired_summary,
        "records": rows,
    }


def reference_design_profile_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise, provenance-bounded companion for a reference profile."""
    if report.get("schema") != REFERENCE_PROFILE_SCHEMA:
        raise ValueError(f"expected profile schema {REFERENCE_PROFILE_SCHEMA!r}")
    overall = report["overall"]
    lines = [
        "# Fixed-Commit Reference Design Profile",
        "",
        "This report describes the released fixed-commit historical open-source reference cohort at the two declared mobile viewports.",
        "It is not a human-authorship detector, visual-preference study, or universal design-quality target.",
        "",
        f"- Source groups: {report['source_groups']}",
        f"- Mobile captures: {report['captures']}",
        f"- Paired 390x844 / 430x932 source groups: {report['paired_mobile_viewport_deltas']['paired_source_groups']}",
        "- Overall medians first average each source group across its two declared mobile captures.",
        "",
        "## Overall Mobile Profile",
        "",
        "| Metric | Source-level median (IQR) |",
        "| --- | --- |",
    ]
    labels = {
        "font_size_p50_px": "Text font-size p50 (px)",
        "font_size_iqr_px": "Text font-size IQR (px)",
        "type_hierarchy_ratio": "Type hierarchy p90 / p50",
        "line_height_p50": "Line-height ratio p50",
        "line_height_iqr": "Line-height ratio IQR",
        "grid_8px_snap_rate": "8px phase-alignment proxy",
        "palette_unique_count": "Unique computed colors",
        "palette_top5_share": "Top-5 color share",
        "neutral_color_share": "Neutral color share (S <= 0.10)",
        "saturation_p50": "Saturation p50",
    }
    for metric in PROFILE_METRICS:
        lines.append(f"| {labels[metric]} | {_metric_cell(overall, metric)} |")
    lines.extend(["", "## By Page Type", "", "| Page type | Sources | Captures | Type hierarchy | Neutral share |", "| --- | ---: | ---: | --- | --- |"])
    for page_type, summary in sorted(report["by_page_type"].items()):
        lines.append(
            "| "
            f"{page_type} | {summary['source_groups']} | {summary['captures']} | "
            f"{_metric_cell(summary, 'type_hierarchy_ratio')} | "
            f"{_metric_cell(summary, 'neutral_color_share')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The reported values are descriptive source-level summaries. Typography and color measurements are browser-computed samples; the 8px value is a left-position phase proxy; and paired viewport differences describe responsive change rather than an aesthetic or accessibility score. The historical cutoff and source-project evidence reduce provenance ambiguity but do not establish individual human authorship or exclude all forms of automated assistance.",
            "",
        ]
    )
    return "\n".join(lines)
