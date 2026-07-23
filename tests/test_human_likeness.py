"""Tests for the leakage-safe WebHumanBench protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.human_likeness import (
    SCHEMA,
    evaluate_human_likeness_benchmark,
    fit_reference_by_page_type,
    human_fit_percentile,
    validate_manifest,
)


def _features(offset: float = 0.0) -> dict[str, object]:
    colors = ["#1f2937", "#f8fafc", "#0f766e", "#d97706"]
    if offset > 0:
        colors.append("#7c3aed")
    return {
        "typography": [16.0 + offset, 18.0 + offset, 20.0 + offset],
        "spacing": [1.4 + offset / 20, 1.6 + offset / 20],
        "grid": [4.0 + offset / 5, 5.0 + offset / 5],
        "color": colors,
        "saturation": [0.35 + offset / 100, 0.45 + offset / 100],
    }


def _record(record_id: str, source: str, split: str, page_type: str, offset: float, model_id: str | None = None, viewport: str = "390x844") -> dict[str, object]:
    record: dict[str, object] = {
        "id": record_id,
        "source": source,
        "split": split,
        "group_id": f"group-{record_id}",
        "page_type": page_type,
        "viewport": viewport,
        "features": _features(offset),
    }
    if model_id:
        record["model_id"] = model_id
    return record


def _manifest() -> dict[str, object]:
    records = [
        _record("h-train-saas-a", "human", "train", "saas_landing", -0.5),
        _record("h-train-saas-b", "human", "train", "saas_landing", 0.5),
        _record("h-train-docs-a", "human", "train", "docs_homepage", -2.3),
        _record("h-train-docs-b", "human", "train", "docs_homepage", 0.3),
        _record("h-dev-saas", "human", "dev", "saas_landing", 0.1),
        _record("h-dev-docs", "human", "dev", "docs_homepage", -1.1),
        _record("h-test-saas", "human", "test", "saas_landing", 0.0),
        _record("h-test-docs", "human", "test", "docs_homepage", -1.0),
        _record("ai-test-saas", "ai", "test", "saas_landing", 8.0, "model-a"),
        _record("ai-test-docs", "ai", "test", "docs_homepage", -7.0, "model-a"),
    ]
    return {
        "schema": SCHEMA,
        "metadata": {
            "benchmark_name": "Synthetic protocol fixture",
            "version": "0",
            "reference_scope": "fixture only",
            "license": "CC0-1.0",
            "required_page_types": ["saas_landing", "docs_homepage"],
            "mobile_test_share_min": 0.75,
        },
        "records": records,
    }


def test_human_likeness_evaluation_separates_synthetic_sources():
    result = evaluate_human_likeness_benchmark(_manifest())
    assert result["test"]["n_rows"] == 4
    assert result["test"]["human_vs_ai_auroc"] == pytest.approx(1.0)
    assert set(result["test"]["by_model"]) == {"model-a"}
    assert result["reference"]["mode"] == "page_type_conditioned"
    assert set(result["reference"]["by_page_type"]) == {"saas_landing", "docs_homepage"}
    assert result["test"]["by_page_type"]["saas_landing"]["human_vs_ai_auroc"] == pytest.approx(1.0)
    assert result["test"]["calibration_scope"]["limited_page_types"] == ["docs_homepage", "saas_landing"]
    assert result["test"]["calibration_scope"]["per_page_type"]["saas_landing"]["max_midrank_percentile_levels"] == 3


def test_type_conditioned_references_preserve_page_type_specific_centers():
    records = validate_manifest(_manifest())
    references = fit_reference_by_page_type(record for record in records if record.split == "train")
    assert references["saas_landing"].means["typography"] != references["docs_homepage"].means["typography"]


def test_zero_variance_train_dimension_is_excluded_from_normalized_scoring():
    manifest = _manifest()
    for record in manifest["records"]:
        if record["split"] == "train" and record["page_type"] == "saas_landing":
            record["features"]["spacing"] = [1.2, 1.2]

    result = evaluate_human_likeness_benchmark(manifest)

    reference = result["reference"]["by_page_type"]["saas_landing"]
    assert reference["inactive_zero_variance_dimensions"] == ["spacing"]
    assert all(
        row["per_dimension"]["spacing"] == 0.0
        for row in result["test"]["rows"]
        if row["page_type"] == "saas_landing"
    )


def test_manifest_rejects_group_leakage_across_splits():
    manifest = _manifest()
    manifest["records"][0]["group_id"] = manifest["records"][4]["group_id"]
    with pytest.raises(ValueError, match="cross splits"):
        validate_manifest(manifest)


def test_manifest_rejects_scoring_group_across_splits_even_with_distinct_lineages():
    manifest = _manifest()
    manifest["records"][0]["group_id"] = "shared-scoring-group"
    manifest["records"][4]["group_id"] = "shared-scoring-group"
    manifest["records"][0]["leakage_group_id"] = "shared-lineage"
    manifest["records"][4]["leakage_group_id"] = "shared-lineage"
    with pytest.raises(ValueError, match="scoring group"):
        validate_manifest(manifest)


def test_manifest_rejects_leakage_lineage_across_splits():
    manifest = _manifest()
    manifest["records"][0]["leakage_group_id"] = "shared-site"
    manifest["records"][4]["leakage_group_id"] = "shared-site"
    with pytest.raises(ValueError, match="leakage_group_id"):
        validate_manifest(manifest)


def test_manifest_allows_model_specific_scoring_groups_with_shared_prompt_lineage():
    manifest = _manifest()
    ai_copy = dict(manifest["records"][-2])
    ai_copy["id"] = "ai-test-saas-model-b"
    ai_copy["group_id"] = "prompt-saas-model-b"
    ai_copy["leakage_group_id"] = "prompt-saas"
    ai_copy["model_id"] = "model-b"
    manifest["records"][-2]["group_id"] = "prompt-saas-model-a"
    manifest["records"][-2]["leakage_group_id"] = "prompt-saas"
    manifest["records"].append(ai_copy)
    result = evaluate_human_likeness_benchmark(manifest)
    assert set(result["test"]["by_model"]) == {"model-a", "model-b"}


def test_manifest_rejects_ai_reference_records():
    manifest = _manifest()
    manifest["records"][0]["source"] = "ai"
    manifest["records"][0]["model_id"] = "model-a"
    with pytest.raises(ValueError, match="train may contain only human"):
        validate_manifest(manifest)


def test_human_fit_percentile_has_midrank_ties():
    assert human_fit_percentile(2.0, [1.0, 2.0, 3.0]) == pytest.approx(50.0)


def test_manifest_rejects_insufficient_mobile_test_coverage():
    manifest = _manifest()
    for record in manifest["records"]:
        if record["split"] == "test":
            record["viewport"] = "1440x900"
    with pytest.raises(ValueError, match="mobile share"):
        validate_manifest(manifest)


def test_manifest_requires_human_and_ai_test_records_in_each_page_type():
    manifest = _manifest()
    manifest["records"] = [
        record
        for record in manifest["records"]
        if record["id"] != "ai-test-docs"
    ]
    with pytest.raises(ValueError, match="docs_homepage"):
        validate_manifest(manifest)


def test_manifest_requires_two_human_train_groups_per_page_type():
    manifest = _manifest()
    manifest["records"] = [
        record
        for record in manifest["records"]
        if record["id"] != "h-train-saas-b"
    ]
    with pytest.raises(ValueError, match="at least two human source groups"):
        validate_manifest(manifest)
