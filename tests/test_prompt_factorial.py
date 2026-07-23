"""Tests for the matched prompt-factorial protocol and analysis."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_prompt_factorial import _paths, _prepare_capture_output_paths

from webmark.prompt_factorial import (
    CONDITIONS,
    PAGE_TYPES,
    analyze_prompt_factorial,
    build_factorial_plan,
    plan_as_dicts,
    prompt_messages,
)


def test_factorial_plan_is_balanced_and_block_matched() -> None:
    plan = build_factorial_plan(model="provider/model", blocks_per_type=3, seed=17)

    assert len(plan) == len(PAGE_TYPES) * len(CONDITIONS) * 3
    assert len({page.page_id for page in plan}) == len(plan)
    for page_type in PAGE_TYPES:
        type_pages = [page for page in plan if page.page_type == page_type]
        assert {page.condition for page in type_pages} == set(CONDITIONS)
        for block_id in {page.block_id for page in type_pages}:
            block = [page for page in type_pages if page.block_id == block_id]
            assert len(block) == len(CONDITIONS)
            assert len({page.scenario for page in block}) == 1
            assert len({page.seed for page in block}) == 1


def test_prompts_change_only_the_declared_factor_clauses() -> None:
    plan = build_factorial_plan(model="provider/model", blocks_per_type=1, seed=17)
    by_condition = {page.condition: prompt_messages(page) for page in plan if page.page_type == PAGE_TYPES[0]}

    for condition, factors in CONDITIONS.items():
        user = by_condition[condition][1]["content"]
        assert ("Use literal CSS values" in user) is factors["literal_css"]
        assert ("modern human-authored website" in user) is factors["human_style"]
        assert "Do not use external assets" in user


def test_capture_preparation_creates_only_runtime_artifact_directories(tmp_path: Path) -> None:
    page = build_factorial_plan(model="provider/model", blocks_per_type=1, seed=17)[0]
    paths = _paths(page, tmp_path / "artifacts")

    _prepare_capture_output_paths(paths)

    assert paths["rendered_html"].parent.is_dir()
    assert paths["screenshot"].parent.is_dir()
    assert paths["computed_features"].parent.is_dir()
    assert not paths["generated_html"].parent.exists()


def test_factorial_analysis_uses_matched_within_type_effects() -> None:
    plan = build_factorial_plan(model="provider/model", blocks_per_type=2, seed=17)
    distances = {"neutral": 10.0, "literal_only": 6.0, "style_only": 8.0, "full": 4.0}
    groups = []
    records = []
    for page_type in PAGE_TYPES:
        for index in range(2):
            groups.append(
                {
                    "group_id": f"historical-{page_type}-{index}",
                    "source": "human",
                    "page_type": page_type,
                    "distance": 1.0,
                    "reference_fit_score": -1.0,
                }
            )
    for page in plan:
        group_id = f"generation-{page.page_id}"
        distance = distances[page.condition]
        groups.append(
            {
                "group_id": group_id,
                "source": "ai",
                "page_type": page.page_type,
                "distance": distance,
                "reference_fit_score": -distance,
            }
        )
        records.append(
            {
                "group_id": group_id,
                "features": {
                    "typography": [14.0, 16.0],
                    "spacing": [1.2, 1.5],
                    "grid": [1.0, 2.0],
                    "color": ["rgb(0, 0, 0)", "rgb(255, 255, 255)"],
                    "saturation": [0.0, 0.0],
                },
            }
        )
    baseline = {"baselines": {"profile_l2_w1": {"groups": groups}}}

    first = analyze_prompt_factorial(
        baseline, records, plan_as_dicts(plan), n_resamples=100, seed=9
    )
    second = analyze_prompt_factorial(
        baseline, records, plan_as_dicts(plan), n_resamples=100, seed=9
    )

    assert first == second
    effects = first["baselines"]["profile_l2_w1"]["paired_factor_effects"]["effects"]
    assert effects["literal_css"]["equal_page_type_paired_closer_fit_rate"] == 1.0
    assert effects["human_style"]["equal_page_type_paired_closer_fit_rate"] == 1.0
    assert effects["literal_css"]["by_page_type"][PAGE_TYPES[0]]["median_distance_delta"] == -4.0
    assert effects["human_style"]["by_page_type"][PAGE_TYPES[0]]["median_distance_delta"] == -2.0
    assert first["feature_diagnostics"]["neutral"]["n_groups"] == len(PAGE_TYPES) * 2
    assert "median_group_spacing_fraction_1_2" in first["feature_diagnostics"]["neutral"]
    assert "pooled_group_median_spacing_fraction_1_2" not in first["feature_diagnostics"]["neutral"]
