"""Tests for frozen WebHumanBench v1 sensitivity diagnostics."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_human_likeness import _manifest

from webmark.benchmark_baselines import evaluate_reference_fit_baselines
from webmark.benchmark_diagnostics import (
    audit_ai_exclusions,
    evaluate_challenge_composition_sensitivity,
    evaluate_feature_dominance,
    evaluate_leave_one_reference_out,
    evaluate_measurement_artifacts,
    evaluate_model_strata,
    evaluate_split_sensitivity,
    summarize_viewport_stability,
)


def test_split_sensitivity_is_deterministic_and_preserves_counts() -> None:
    first = evaluate_split_sensitivity(_manifest(), n_alternative_splits=4, seed=7)
    second = evaluate_split_sensitivity(_manifest(), n_alternative_splits=4, seed=7)
    assert first == second
    assert first["n_alternative_splits"] == 4
    assert len({run["assignment_sha256"] for run in first["runs"]}) == 4
    assert first["counts_by_page_type"]["saas_landing"] == {
        "train": 2,
        "dev": 1,
        "test": 1,
    }
    for baseline in first["baseline_auroc_summary"].values():
        assert 0 <= baseline["min"] <= baseline["max"] <= 1


def test_model_strata_reuse_the_same_historical_groups() -> None:
    baselines = evaluate_reference_fit_baselines(
        _manifest(), n_resamples=10, min_groups_for_ci=100
    )
    result = evaluate_model_strata(
        baselines, n_resamples=20, seed=9, min_groups_for_ci=2
    )
    assert result["n_recorded_model_identifiers"] == 1
    assert result["n_shared_historical_test_groups"] == 2
    assert set(result["baselines"]) == set(baselines["baselines"])
    for baseline in result["baselines"].values():
        assert baseline["by_model"]["model-a"]["status"] == "reportable"


def test_challenge_composition_reports_macro_and_leave_one_model_out() -> None:
    baselines = evaluate_reference_fit_baselines(
        _manifest(), n_resamples=10, min_groups_for_ci=100
    )

    result = evaluate_challenge_composition_sensitivity(baselines)

    assert result["primary_endpoint"] == "equal_page_type_macro_auroc"
    assert result["n_recorded_model_identifiers"] == 1
    for baseline in result["baselines"].values():
        assert 0 <= baseline["equal_page_type_macro_auroc"] <= 1
        assert 0 <= baseline["pair_weighted_within_type_auroc"] <= 1
        assert 0 <= baseline["pooled_cross_type_auroc"] <= 1
        leave_one = baseline["leave_one_model_out_type_macro_auroc"]
        assert leave_one["by_excluded_model"] == {}
        assert leave_one["summary"]["status"] == "not_applicable"


def test_leave_one_reference_out_uses_every_historical_source() -> None:
    first = evaluate_leave_one_reference_out(_manifest(), n_resamples=20, seed=11)
    second = evaluate_leave_one_reference_out(_manifest(), n_resamples=20, seed=11)
    assert first == second
    assert first["n_historical_reference_sources"] == 8
    for baseline in first["baselines"].values():
        assert len(baseline["sources"]) == 8
        summary = baseline["summary"]
        assert summary["n_reference_sources"] == 8
        assert 0 <= summary["mean"] <= 1
        assert 0 <= summary["ci_95"][0] <= summary["ci_95"][1] <= 1
        assert {row["n_training_reference_sources"] for row in baseline["sources"]} == {3}


def test_feature_dominance_reports_group_and_weighted_shares() -> None:
    scored = {
        "test": {
            "groups": [
                {
                    "source": "human",
                    "page_type": "saas_landing",
                    "per_dimension": {
                        "typography": 4,
                        "spacing": 1,
                        "grid": 1,
                        "color": 1,
                        "saturation": 1,
                    },
                },
                {
                    "source": "ai",
                    "page_type": "saas_landing",
                    "per_dimension": {
                        "typography": 1,
                        "spacing": 1,
                        "grid": 1,
                        "color": 6,
                        "saturation": 1,
                    },
                },
            ]
        }
    }
    result = evaluate_feature_dominance(
        scored, n_resamples=10, seed=3, min_groups_for_ci=1
    )
    assert result["overall"]["n_groups"] == 2
    assert result["overall"]["dominant_dimension"]["typography"]["count"] == 1
    assert result["overall"]["dominant_dimension"]["color"]["count"] == 1
    assert sum(result["overall"]["score_weighted_share"].values()) == pytest.approx(1.0)
    assert result["feature_ablation_auroc"]["full_profile"]["point_estimate"] == pytest.approx(1.0)
    assert "without_color_and_spacing" in result["feature_ablation_auroc"]


def test_viewport_stability_identifies_only_nonzero_median() -> None:
    profile = {
        "by_viewport": {"390x844": {}, "430x932": {}},
        "paired_mobile_viewport_deltas": {
            "paired_source_groups": 3,
            "absolute_delta": {
                "font": {"median": 0.0, "iqr": 0.0, "n": 3},
                "phase": {"median": 0.02, "iqr": 0.1, "n": 3},
            },
        },
    }
    result = summarize_viewport_stability(profile)
    assert result["n_zero_median_absolute_delta"] == 1
    assert result["largest_median_absolute_delta"] == {"metric": "phase", "value": 0.02}


def test_measurement_artifact_audit_reports_proxy_saturation_and_density() -> None:
    manifest = copy.deepcopy(_manifest())
    result = evaluate_measurement_artifacts(manifest)

    reference = result["by_cohort"]["historical_reference"]
    challenge = result["by_cohort"]["generated_challenge"]
    assert reference["n_groups"] == 8
    assert challenge["n_groups"] == 2
    assert 0 <= reference["spacing_value_1_2"]["pooled_sample_fraction"] <= 1
    assert 0 <= challenge["spacing_value_1_2"]["pooled_sample_fraction"] <= 1
    assert "color_unique_vs_color_sample_count_spearman" in reference


def test_ai_exclusion_audit_closes_source_run_without_score_selection() -> None:
    manifest = copy.deepcopy(_manifest())
    ai_record = next(record for record in manifest["records"] if record["source"] == "ai")
    ai_record["id"] = "ai-page-a"
    ai_record["provenance"] = {"prompt_id": "page-a"}
    manifest["records"] = [
        record
        for record in manifest["records"]
        if record["source"] != "ai" or record is ai_record
    ]
    exclusion_ledger = {
        "schema": "webhumanbench_v1_ai_exclusion_ledger_v1",
        "retained_pages": 1,
        "excluded_pages": 1,
        "selection_uses_reference_fit_score": False,
        "exclusions": [
            {
                "page_id": "page-b",
                "page_type": "saas_landing",
                "model_id": "model-b",
                "error": (
                    "'page-b' requested resources outside its archived HTML: "
                    "'https://example.invalid/image.png'"
                ),
            }
        ],
    }
    source_run_ledger = {
        "schema": "webhumanbench_v1_ai_source_run_ledger_v1",
        "planned_pages": 2,
        "records": [
            {"id": "page-a", "page_type": "saas_landing", "model_id": "model-a"},
            {"id": "page-b", "page_type": "saas_landing", "model_id": "model-b"},
        ]
    }
    result = audit_ai_exclusions(manifest, exclusion_ledger, source_run_ledger)
    assert result["closure_verified"] is True
    assert result["selection_uses_reference_fit_score"] is False
    assert result["by_page_type"]["saas_landing"] == {
        "planned": 2,
        "retained": 1,
        "excluded": 1,
    }
    assert result["by_model"]["model-b"] == {
        "planned": 1,
        "retained": 0,
        "excluded": 1,
    }
