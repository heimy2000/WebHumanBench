"""Tests for scope-safe candidate-cohort design profiling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_public_release import _historical_evidence_release_fixture

from webmark.design_profile import (
    PROFILE_METRICS,
    PROFILE_SCHEMA,
    REFERENCE_PROFILE_SCHEMA,
    build_candidate_design_profile,
    build_reference_design_profile,
    candidate_design_profile_markdown,
    profile_features,
    reference_design_profile_markdown,
    summarize_profiles,
)
from webmark.release import canonical_json_sha256


def _features() -> dict[str, list[float] | list[str]]:
    return {
        "typography": [12.0, 16.0, 16.0, 24.0, 32.0],
        "spacing": [1.2, 1.4, 1.5, 1.6, 1.7],
        "grid": [0.0, 1.0, 2.01, 3.0, 4.5],
        "color": ["rgb(0, 0, 0)", "rgb(0, 0, 0)", "rgb(255, 0, 0)", "rgb(255, 255, 255)"],
        "saturation": [0.0, 0.0, 1.0, 0.0],
    }


def _probe(task_id: str, viewport: str, page_type: str = "saas_landing") -> dict[str, object]:
    return {
        "task_id": task_id,
        "repository": f"example/{task_id}",
        "commit_sha": "a" * 40,
        "page_type": page_type,
        "viewport": viewport,
        "probe_status": "captured",
        "rendered_url": "https://example.org/",
        "screenshot_sha256": "b" * 64,
        "feature_sha256": "c" * 64,
        "features": _features(),
    }


def test_profile_features_reports_interpretable_measurements() -> None:
    profile = profile_features(_features())
    assert profile["font_size_p50_px"] == pytest.approx(16.0)
    assert profile["type_hierarchy_ratio"] == pytest.approx(1.8)
    assert profile["palette_unique_count"] == 3
    assert profile["neutral_color_share"] == pytest.approx(0.75)
    assert profile["grid_8px_snap_rate"] == pytest.approx(0.8)


def test_candidate_profile_requires_retained_features() -> None:
    payload = {
        "schema": "webmark_open_reference_candidate_mobile_probe_v1",
        "probes": [{**_probe("source-a", "390x844"), "features": None}],
    }
    with pytest.raises(ValueError, match="retain-features"):
        build_candidate_design_profile(payload)


def test_candidate_profile_groups_two_viewports_and_marks_candidate_scope() -> None:
    payload = {
        "schema": "webmark_open_reference_candidate_mobile_probe_v1",
        "probes": [
            _probe("source-a", "390x844"),
            _probe("source-a", "430x932"),
            _probe("source-b", "390x844", "docs_homepage"),
            _probe("source-b", "430x932", "docs_homepage"),
        ],
    }
    report = build_candidate_design_profile(payload)
    assert report["schema"] == PROFILE_SCHEMA
    assert report["data_status"] == "unverified_current_remote_candidate_cohort"
    assert report["source_groups"] == 2
    assert report["captures"] == 4
    assert report["paired_mobile_viewport_deltas"]["paired_source_groups"] == 2
    markdown = candidate_design_profile_markdown(report)
    assert "not evidence of human authorship" in markdown
    assert "historical commit parity" in markdown


def test_reference_profile_requires_complete_fixed_commit_capture(tmp_path: Path) -> None:
    sources, captures, _benchmark = _historical_evidence_release_fixture(tmp_path)

    report = build_reference_design_profile(sources, captures)

    assert report["schema"] == REFERENCE_PROFILE_SCHEMA
    assert report["data_status"] == "fixed_commit_historical_open_source_reference_cohort"
    assert report["source_groups"] == 8
    assert report["captures"] == 16
    assert report["paired_mobile_viewport_deltas"]["paired_source_groups"] == 8
    assert report["source_manifest_sha256"] == canonical_json_sha256(sources)
    markdown = reference_design_profile_markdown(report)
    assert "not a human-authorship detector" in markdown
    assert "historical cutoff" in markdown


def test_reference_profile_rejects_missing_primary_capture(tmp_path: Path) -> None:
    sources, captures, _benchmark = _historical_evidence_release_fixture(tmp_path)
    captures["records"] = [
        record
        for record in captures["records"]
        if record["id"] != "saas-train-a@430x932"
    ]

    with pytest.raises(ValueError, match="lacks reference capture"):
        build_reference_design_profile(sources, captures)


def test_summary_aggregates_viewports_before_source_level_statistics() -> None:
    base = {metric: 0.0 for metric in PROFILE_METRICS}
    summary = summarize_profiles(
        [
            {"source_group": "source-a", "profile": {**base, "font_size_p50_px": 0.0}},
            {"source_group": "source-a", "profile": {**base, "font_size_p50_px": 10.0}},
            {"source_group": "source-b", "profile": {**base, "font_size_p50_px": 4.0}},
        ]
    )

    assert summary["source_groups"] == 2
    assert summary["captures"] == 3
    assert summary["aggregation"] == "source_group_mean_over_retained_captures_v1"
    assert summary["metrics"]["font_size_p50_px"] == {"n": 2, "median": 4.5, "iqr": 0.5}
