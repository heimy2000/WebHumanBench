"""Tests for one-class visual-embedding benchmark baselines."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from webmark.visual_embedding_baseline import (  # noqa: E402
    BASELINES,
    evaluate_visual_embedding_baselines,
)


def _fixture() -> tuple[dict[str, object], dict[str, list[float]]]:
    records = []
    embeddings = {}
    for page_type in ("saas_landing", "docs_homepage"):
        for index, split in enumerate(("train", "train", "dev", "test")):
            group_id = f"reference-{page_type}-{index}"
            records.append(
                {
                    "id": group_id,
                    "group_id": group_id,
                    "source": "human",
                    "split": split,
                    "page_type": page_type,
                }
            )
            embeddings[group_id] = [1.0, 0.01 * index]
        for index in range(3):
            group_id = f"challenge-{page_type}-{index}"
            records.append(
                {
                    "id": group_id,
                    "group_id": group_id,
                    "source": "ai",
                    "split": "test",
                    "page_type": page_type,
                    "model_id": "provider/model",
                }
            )
            embeddings[group_id] = [0.01 * index, 1.0]
    manifest = {
        "schema": "fixture",
        "metadata": {"version": "fixture"},
        "records": records,
    }
    return manifest, embeddings


def test_visual_embedding_baselines_are_group_level_and_deterministic() -> None:
    manifest, embeddings = _fixture()

    first = evaluate_visual_embedding_baselines(
        manifest,
        embeddings,
        n_resamples=100,
        n_alternative_splits=4,
        seed=11,
    )
    second = evaluate_visual_embedding_baselines(
        manifest,
        embeddings,
        n_resamples=100,
        n_alternative_splits=4,
        seed=11,
    )

    assert first == second
    assert first["n_groups"] == 14
    assert first["embedding_dimension"] == 2
    for baseline in BASELINES:
        result = first["baselines"][baseline]
        assert result["frozen_type_macro"]["point_estimate"] == 1.0
        assert result["frozen_type_macro"]["n_human_groups"] == 2
        assert result["frozen_type_macro"]["n_ai_groups"] == 6
        assert result["leave_one_reference_out"]["summary"]["mean"] == 1.0
        assert result["source_split_sensitivity"]["median"] == 1.0


def test_recorded_path_resolves_relative_cli_paths_and_preserves_external_paths(
    monkeypatch, tmp_path: Path
) -> None:
    from run_visual_embedding_baseline import _recorded_path

    monkeypatch.chdir(ROOT)

    assert _recorded_path(Path("results/example.json")) == "results/example.json"
    assert _recorded_path(tmp_path / "external.json") == (tmp_path / "external.json").as_posix()


def test_analysis_only_screenshot_rows_use_published_digests_without_raw_files(tmp_path: Path) -> None:
    from run_visual_embedding_baseline import _screenshot_rows

    manifest = {
        "records": [
            {
                "id": "reference-1",
                "group_id": "reference-1",
                "source": "human",
                "split": "test",
                "page_type": "docs_homepage",
                "provenance": {"capture_id": "capture-1"},
            },
            {
                "id": "challenge-1",
                "group_id": "challenge-1",
                "source": "ai",
                "split": "test",
                "page_type": "docs_homepage",
                "provenance": {
                    "artifacts": {"screenshot": {"path": "challenge.png", "sha256": "challenge-digest"}}
                },
            },
        ]
    }
    capture_ledger = {
        "records": [
            {
                "id": "capture-1",
                "artifacts": {"screenshot": {"path": "reference.png", "sha256": "reference-digest"}},
            }
        ]
    }

    rows = _screenshot_rows(manifest, capture_ledger, capture_root=tmp_path, verify_files=False)

    assert [(row["group_id"], row["screenshot_sha256"]) for row in rows] == [
        ("challenge-1", "challenge-digest"),
        ("reference-1", "reference-digest"),
    ]
