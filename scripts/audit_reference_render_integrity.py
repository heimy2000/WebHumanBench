#!/usr/bin/env python3
"""Screen fixed-source captures for blank or error-like rendered states.

This deterministic screen checks capture integrity, not human authorship,
page-type validity, visual preference, accessibility, or design quality.  It
exists to prevent a syntactically complete feature vector from admitting an
obvious blank, error, or externally altered page into a candidate cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.open_reference import (  # noqa: E402
    PRIMARY_MOBILE_VIEWPORTS,
    validate_open_reference_manifest,
)
from webmark.release import canonical_json_sha256, sha256_file  # noqa: E402

CAPTURE_SCHEMA = "webmark_open_mobile_capture_v2"
SCREEN_SCHEMA = "webmark_reference_render_integrity_screen_v1"
ERROR_MARKERS = (
    "application error",
    "cannot get /",
    "chunkloaderror",
    "internal server error",
    "page not found",
    "runtime error",
    "something went wrong",
    "this site can't be reached",
    "unhandled exception",
)
UNRENDERED_TEMPLATE_MARKERS = ("<%", "%>", "{{ .", "{{range", "{{ range", "{%", "%}")
MIN_VISIBLE_TEXT_CHARS = 80
MIN_FEATURE_SAMPLES = 2
MIN_SCREENSHOT_CHANNEL_STD = 1.0


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "template", "noscript"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "template", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def _visible_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _safe_artifact(root: Path, raw: Any, field: str) -> Path:
    item = raw if isinstance(raw, Mapping) else None
    if item is None:
        raise ValueError(f"{field} must be an artifact object")
    path = PurePosixPath(str(item.get("path", "")))
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in str(path):
        raise ValueError(f"{field}.path must be a safe relative path")
    digest = str(item.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    resolved = root / path
    if not resolved.is_file():
        raise ValueError(f"{field}.path does not exist")
    if sha256_file(resolved) != digest:
        raise ValueError(f"{field}.sha256 does not match the artifact")
    return resolved


def _screenshot_channel_std(path: Path) -> float:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        variance = ImageStat.Stat(rgb).var
    return math.sqrt(sum(float(value) for value in variance) / len(variance))


def _screen_record(record: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    source_id = str(record.get("source_id", ""))
    viewport = str(record.get("viewport", ""))
    output: dict[str, Any] = {
        "source_id": source_id,
        "viewport": viewport,
        "capture_id": record.get("id"),
        "status": "pass",
        "reasons": [],
    }
    try:
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("capture lacks artifacts")
        html_path = _safe_artifact(artifact_root, artifacts.get("html"), "html")
        screenshot_path = _safe_artifact(artifact_root, artifacts.get("screenshot"), "screenshot")
        html_digest = str(record.get("capture_html_sha256", ""))
        if html_digest != sha256_file(html_path):
            raise ValueError("capture_html_sha256 does not match HTML artifact")
        html = html_path.read_text(encoding="utf-8", errors="replace")
        text = _visible_text(html)
        lowered = text.lower()
        features = record.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("capture lacks features")
        counts = {
            field: len(features.get(field, [])) if isinstance(features.get(field), list) else 0
            for field in ("typography", "spacing", "grid", "color", "saturation")
        }
        output["visible_text_chars"] = len(text)
        output["feature_sample_counts"] = counts
        output["screenshot_channel_std"] = _screenshot_channel_std(screenshot_path)
        if len(text) < MIN_VISIBLE_TEXT_CHARS:
            output["reasons"].append("visible_text_below_minimum")
        markers = [marker for marker in ERROR_MARKERS if marker in lowered]
        if markers:
            output["reasons"].append("error_marker:" + markers[0])
        template_markers = [marker for marker in UNRENDERED_TEMPLATE_MARKERS if marker in html]
        if template_markers:
            output["reasons"].append("unrendered_template_syntax:" + template_markers[0])
        for field, count in counts.items():
            if count < MIN_FEATURE_SAMPLES:
                output["reasons"].append(f"insufficient_{field}_samples")
        if output["screenshot_channel_std"] < MIN_SCREENSHOT_CHANNEL_STD:
            output["reasons"].append("near_uniform_screenshot")
    except (OSError, ValueError, Image.UnidentifiedImageError) as exc:
        output["reasons"].append(f"artifact_or_parse_error:{exc}")
    if output["reasons"]:
        output["status"] = "fail"
    return output


def audit_reference_render_integrity(
    source_manifest: Mapping[str, Any], capture_ledger: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Return source-level candidate-screen evidence for a fixed capture ledger."""
    sources = validate_open_reference_manifest(source_manifest)
    if capture_ledger.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"capture ledger schema must be {CAPTURE_SCHEMA!r}")
    if capture_ledger.get("source_manifest_sha256") != canonical_json_sha256(source_manifest):
        raise ValueError("capture ledger source_manifest_sha256 does not match source manifest")
    records = capture_ledger.get("records")
    failures = capture_ledger.get("failures", [])
    if not isinstance(records, list) or not isinstance(failures, list):
        raise ValueError("capture ledger requires records and failures lists")
    source_by_id = {source.id: source for source in sources}
    screened: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            continue
        source_id = str(raw_record.get("source_id", ""))
        viewport = str(raw_record.get("viewport", ""))
        if source_id not in source_by_id or viewport not in PRIMARY_MOBILE_VIEWPORTS:
            continue
        key = (source_id, viewport)
        if key in screened:
            raise ValueError(f"capture ledger has duplicate source/viewport pair {key!r}")
        screened[key] = _screen_record(raw_record, artifact_root)
    failed_by_source: dict[str, list[str]] = defaultdict(list)
    for raw_failure in failures:
        if isinstance(raw_failure, Mapping):
            source_id = str(raw_failure.get("source_id", ""))
            if source_id in source_by_id:
                failed_by_source[source_id].append(str(raw_failure.get("error", "capture_failure")))

    source_rows: list[dict[str, Any]] = []
    for source in sources:
        captures: list[dict[str, Any]] = []
        reasons = list(failed_by_source.get(source.id, []))
        for viewport in PRIMARY_MOBILE_VIEWPORTS:
            row = screened.get((source.id, viewport))
            if row is None:
                reasons.append(f"missing_capture:{viewport}")
            else:
                captures.append(row)
                reasons.extend(str(reason) for reason in row["reasons"])
        source_rows.append(
            {
                "source_id": source.id,
                "page_type": source.page_type,
                "status": "pass" if not reasons else "fail",
                "reasons": sorted(set(reasons)),
                "captures": captures,
            }
        )
    passed = sum(row["status"] == "pass" for row in source_rows)
    return {
        "schema": SCREEN_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(source_manifest),
        "capture_ledger_sha256": canonical_json_sha256(capture_ledger),
        "artifact_root": str(artifact_root),
        "thresholds": {
            "min_visible_text_chars": MIN_VISIBLE_TEXT_CHARS,
            "min_feature_samples": MIN_FEATURE_SAMPLES,
            "min_screenshot_channel_std": MIN_SCREENSHOT_CHANNEL_STD,
        },
        "status": "pass" if passed == len(source_rows) else "partial_candidate_screen",
        "n_sources": len(source_rows),
        "n_passed": passed,
        "n_failed": len(source_rows) - passed,
        "sources": source_rows,
        "note": (
            "This deterministic render-integrity screen rejects blank/error-like captured states. It does not "
            "verify individual human authorship, page-type correctness, aesthetic preference, accessibility, "
            "or design quality."
        ),
    }


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--capture-ledger", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_reference_render_integrity(
        _load_object(args.source_manifest),
        _load_object(args.capture_ledger),
        artifact_root=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"render-integrity screen: {report['n_passed']}/{report['n_sources']} sources passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
