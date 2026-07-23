#!/usr/bin/env python3
"""Compare two browser captures of the frozen prompt-factorial pages."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.prompt_factorial import PROMPT_FACTORIAL_SCHEMA
from webmark.release import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("typography", "spacing", "grid", "color", "saturation")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _resolve_artifact_root(result: dict[str, Any], result_path: Path) -> Path:
    value = Path(str(result.get("artifact_root", "")))
    if value.is_absolute():
        return value
    candidates = (ROOT / value, result_path.parent / value)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"cannot resolve artifact root {value!s} for {result_path}")


def _records(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if result.get("schema") != PROMPT_FACTORIAL_SCHEMA or result.get("status") != "complete":
        raise ValueError("browser-patch comparison requires two complete prompt-factorial results")
    rows = result.get("records")
    if not isinstance(rows, list):
        raise ValueError("prompt-factorial result requires a records array")
    records = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    if len(records) != len(rows):
        raise ValueError("prompt-factorial result has duplicate or malformed records")
    return records


def _numeric_feature_delta(left: list[Any], right: list[Any]) -> tuple[bool, float]:
    if len(left) != len(right):
        return False, math.inf
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)]
    maximum = max(differences, default=0.0)
    return maximum == 0.0, maximum


def _pixel_delta(left: Path, right: Path) -> tuple[bool, float, int]:
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError as exc:  # pragma: no cover - optional browser dependency
        raise RuntimeError('Install browser dependencies with: pip install -e ".[browser]"') from exc
    with Image.open(left) as left_image, Image.open(right) as right_image:
        left_rgb = left_image.convert("RGB")
        right_rgb = right_image.convert("RGB")
        if left_rgb.size != right_rgb.size:
            return False, math.inf, 255
        difference = ImageChops.difference(left_rgb, right_rgb)
        extrema = difference.getextrema()
        max_channel = max(channel_max for _, channel_max in extrema)
        mean_channel = sum(ImageStat.Stat(difference).mean) / 3.0
        return max_channel == 0, mean_channel, max_channel


def compare_results(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_path: Path,
    right_path: Path,
) -> dict[str, Any]:
    left_records = _records(left)
    right_records = _records(right)
    if set(left_records) != set(right_records):
        raise ValueError("capture page IDs differ between browser patches")
    for field in ("benchmark_manifest_sha256",):
        if left.get(field) != right.get(field):
            raise ValueError(f"capture results disagree on {field}")
    if left.get("source_run", {}).get("sha256") != right.get("source_run", {}).get("sha256"):
        raise ValueError("capture results do not share the same frozen source run")

    left_root = _resolve_artifact_root(left, left_path)
    right_root = _resolve_artifact_root(right, right_path)
    changed_by_feature = {feature: [] for feature in FEATURES}
    max_abs_delta = {feature: 0.0 for feature in FEATURES if feature != "color"}
    exact_screenshots = 0
    pixel_changed_ids: list[str] = []
    max_mean_abs_channel_delta = 0.0
    max_channel_delta = 0

    for page_id in sorted(left_records):
        left_record = left_records[page_id]
        right_record = right_records[page_id]
        for key in ("page_type", "condition", "block_id", "group_id"):
            if left_record.get(key) != right_record.get(key):
                raise ValueError(f"capture metadata changed for {page_id}: {key}")
        left_artifacts = left_record["provenance"]["artifacts"]
        right_artifacts = right_record["provenance"]["artifacts"]
        if left_artifacts["generated_html"]["sha256"] != right_artifacts["generated_html"]["sha256"]:
            raise ValueError(f"generated HTML changed for {page_id}")
        for feature in FEATURES:
            left_values = left_record["features"][feature]
            right_values = right_record["features"][feature]
            if feature == "color":
                exact = left_values == right_values
            else:
                exact, maximum = _numeric_feature_delta(left_values, right_values)
                max_abs_delta[feature] = max(max_abs_delta[feature], maximum)
            if not exact:
                changed_by_feature[feature].append(page_id)

        left_png = left_root / left_artifacts["screenshot"]["path"]
        right_png = right_root / right_artifacts["screenshot"]["path"]
        for path, artifact in (
            (left_png, left_artifacts["screenshot"]),
            (right_png, right_artifacts["screenshot"]),
        ):
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"screenshot integrity failed for {path}")
        exact, mean_delta, channel_delta = _pixel_delta(left_png, right_png)
        if exact:
            exact_screenshots += 1
        else:
            pixel_changed_ids.append(page_id)
        max_mean_abs_channel_delta = max(max_mean_abs_channel_delta, mean_delta)
        max_channel_delta = max(max_channel_delta, channel_delta)

    baseline_exact = canonical_json_sha256(left["reference_fit_baselines"]) == canonical_json_sha256(
        right["reference_fit_baselines"]
    )
    analysis_exact = canonical_json_sha256(left["analysis"]) == canonical_json_sha256(right["analysis"])
    all_features_exact = all(not page_ids for page_ids in changed_by_feature.values())
    return {
        "schema": "webmark_prompt_factorial_browser_patch_audit_v1",
        "status": "pass",
        "inputs": {
            "left": {
                "path": str(left_path),
                "sha256": sha256_file(left_path),
                "browser": left["capture"]["browser"],
                "artifact_root": str(left_root),
            },
            "right": {
                "path": str(right_path),
                "sha256": sha256_file(right_path),
                "browser": right["capture"]["browser"],
                "artifact_root": str(right_root),
            },
            "same_source_run_sha256": left["source_run"]["sha256"],
            "same_benchmark_manifest_sha256": left["benchmark_manifest_sha256"],
        },
        "summary": {
            "n_pages": len(left_records),
            "all_computed_features_exact": all_features_exact,
            "exact_png_pixels": exact_screenshots,
            "changed_png_pixels": len(pixel_changed_ids),
            "baseline_object_exact": baseline_exact,
            "analysis_object_exact": analysis_exact,
        },
        "feature_comparison": {
            feature: {
                "changed_pages": len(page_ids),
                "changed_page_ids": page_ids,
                **({"max_absolute_value_delta": max_abs_delta[feature]} if feature != "color" else {}),
            }
            for feature, page_ids in changed_by_feature.items()
        },
        "pixel_comparison": {
            "changed_page_ids": pixel_changed_ids,
            "max_page_mean_absolute_channel_delta_0_255": max_mean_abs_channel_delta,
            "max_channel_delta_0_255": max_channel_delta,
        },
        "claim_boundary": (
            "This paired audit measures sensitivity to the two named Chromium patch builds for the "
            "same 120 frozen HTML documents and capture settings. It is not cross-engine or long-term "
            "browser-version validation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = _load_json(args.left)
    right = _load_json(args.right)
    result = compare_results(left, right, left_path=args.left, right_path=args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "browser-patch audit: "
        f"features_exact={result['summary']['all_computed_features_exact']} "
        f"pixels_exact={result['summary']['exact_png_pixels']}/{result['summary']['n_pages']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
