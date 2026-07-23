"""Tests for leakage-safe reference-only benchmark baselines."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_human_likeness import _manifest

from webmark.benchmark_baselines import (
    BASELINE_NAMES,
    _bootstrap_auc,
    _stratified_bootstrap_auc,
    evaluate_reference_fit_baselines,
)


def test_baseline_suite_uses_only_train_reference_groups() -> None:
    original = _manifest()
    first = evaluate_reference_fit_baselines(original, n_resamples=100, min_groups_for_ci=10)
    modified = copy.deepcopy(original)
    for record in modified["records"]:
        if record["split"] == "dev":
            record["features"]["typography"] = [9_999.0]
            record["features"]["spacing"] = [99.0]
            record["features"]["grid"] = [99.0]
            record["features"]["color"] = ["#000000"]
            record["features"]["saturation"] = [0.0]
    second = evaluate_reference_fit_baselines(modified, n_resamples=100, min_groups_for_ci=10)
    assert set(first["baselines"]) == set(BASELINE_NAMES)
    assert first["baselines"] == second["baselines"]
    assert first["fit_split"] == "human_train_only"


def test_small_pilot_refuses_to_report_bootstrap_ci() -> None:
    rows = [
        {"source": "human", "reference_fit_score": 1.0},
        {"source": "human", "reference_fit_score": 0.8},
        {"source": "ai", "reference_fit_score": 0.2},
        {"source": "ai", "reference_fit_score": 0.1},
    ]
    result = _bootstrap_auc(rows, n_resamples=100, seed=7, min_groups_for_ci=3)
    assert result["status"] == "not_reportable"
    assert result["point_estimate"] == pytest.approx(1.0)


def test_bootstrap_ci_is_deterministic_when_group_counts_are_adequate() -> None:
    rows = [
        {"source": "human", "reference_fit_score": score}
        for score in (0.9, 0.8, 0.7, 0.6)
    ] + [
        {"source": "ai", "reference_fit_score": score}
        for score in (0.4, 0.3, 0.2, 0.1)
    ]
    first = _bootstrap_auc(rows, n_resamples=200, seed=7, min_groups_for_ci=4)
    second = _bootstrap_auc(rows, n_resamples=200, seed=7, min_groups_for_ci=4)
    assert first == second
    assert first["status"] == "reportable"
    assert first["ci_95"] == pytest.approx([1.0, 1.0])


def test_type_macro_auc_does_not_compare_scores_across_page_types() -> None:
    rows = [
        {"source": "human", "page_type": "a", "reference_fit_score": 1.0},
        {"source": "ai", "page_type": "a", "reference_fit_score": 0.0},
        {"source": "human", "page_type": "b", "reference_fit_score": -100.0},
        {"source": "ai", "page_type": "b", "reference_fit_score": -101.0},
    ]

    primary = _stratified_bootstrap_auc(
        rows, n_resamples=50, seed=7, min_groups_for_ci=2
    )
    pooled = _bootstrap_auc(rows, n_resamples=50, seed=7, min_groups_for_ci=2)

    assert primary["point_estimate"] == pytest.approx(1.0)
    assert primary["pair_weighted_within_type_point_estimate"] == pytest.approx(1.0)
    assert pooled["point_estimate"] == pytest.approx(0.75)


def test_baselines_report_and_ignore_zero_variance_train_dimensions() -> None:
    manifest = _manifest()
    for record in manifest["records"]:
        if record["split"] == "train" and record["page_type"] == "saas_landing":
            record["features"]["spacing"] = [1.2, 1.2]

    result = evaluate_reference_fit_baselines(manifest, n_resamples=100, min_groups_for_ci=10)

    assert result["reference_scale_diagnostics"]["saas_landing"] == {
        "active_dimensions": ["typography", "grid", "color", "saturation"],
        "inactive_zero_variance_dimensions": ["spacing"],
    }
