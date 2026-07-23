"""Tests for the paired prompt-factorial browser-patch audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_prompt_factorial_browser_patches import compare_results

from webmark.prompt_factorial import PROMPT_FACTORIAL_SCHEMA
from webmark.release import sha256_file


def _result(root: Path, browser: str) -> dict[str, object]:
    screenshot = root / "screenshots" / "page.png"
    screenshot.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (255, 255, 255)).save(screenshot)
    return {
        "schema": PROMPT_FACTORIAL_SCHEMA,
        "status": "complete",
        "artifact_root": str(root),
        "benchmark_manifest_sha256": "benchmark",
        "source_run": {"sha256": "source-run"},
        "capture": {"browser": browser},
        "analysis": {"value": 1},
        "reference_fit_baselines": {"value": 2},
        "records": [
            {
                "id": "ai-page",
                "group_id": "generation-page",
                "page_type": "saas_landing",
                "condition": "neutral",
                "block_id": "saas_landing:00",
                "features": {
                    "typography": [16.0],
                    "spacing": [1.2],
                    "grid": [1.0],
                    "color": ["rgb(0, 0, 0)"],
                    "saturation": [0.0],
                },
                "provenance": {
                    "artifacts": {
                        "generated_html": {"path": "generated_html/page.html", "sha256": "html"},
                        "screenshot": {
                            "path": "screenshots/page.png",
                            "sha256": sha256_file(screenshot),
                        },
                    }
                },
            }
        ],
    }


def test_browser_patch_comparison_reports_exact_capture(tmp_path: Path) -> None:
    left = _result(tmp_path / "left", "150.0.7871.115")
    right = _result(tmp_path / "right", "150.0.7871.129")
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")

    result = compare_results(
        copy.deepcopy(left),
        copy.deepcopy(right),
        left_path=left_path,
        right_path=right_path,
    )

    assert result["status"] == "pass"
    assert result["summary"]["all_computed_features_exact"] is True
    assert result["summary"]["exact_png_pixels"] == 1
    assert result["summary"]["baseline_object_exact"] is True
    assert result["summary"]["analysis_object_exact"] is True
