"""Browser-scored bounded CSS alignment on a frozen WebHumanBench split.

The controller selects CSS operator chains using the same computed-style
feature extractor used by the benchmark, while fitting its reference only on
historical-reference train groups.  This closes the implementation gap between
the released benchmark and the optional alignment track.  It does not turn a
score reduction into an aesthetic or authorship claim.
"""
from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .beam_search import BeamSearchConfig
from .bias import BiasScorer
from .contrast import static_low_contrast_exposure
from .features import FEATURE_NAMES, PageFeatures
from .human_likeness import fit_reference_by_page_type, validate_manifest
from .operators import OperatorName, apply_operator
from .release import sha256_file

ALIGNMENT_SCHEMA = "webmark_browser_scored_alignment_v1"


def computed_style_feature_script() -> str:
    """Return the deterministic browser feature extractor used by v0.2 captures."""
    return """
    () => {
      const visible = (el) => {
        const box = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return box.width > 0 && box.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
      };
      const hasOwnText = (el) => Array.from(el.childNodes).some(
        (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim().length > 0
      );
      const parseRgb = (value) => {
        const match = value.match(/rgba?\\(([^)]+)\\)/);
        if (!match) return null;
        const parts = match[1].split(',').map((part) => parseFloat(part.trim()));
        if (parts.length < 3 || (parts.length > 3 && parts[3] === 0)) return null;
        return parts.slice(0, 3).map((part) => Math.max(0, Math.min(255, Math.round(part))));
      };
      const rgbString = (rgb) => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
      const saturation = (rgb) => {
        const values = rgb.map((value) => value / 255);
        const high = Math.max(...values);
        const low = Math.min(...values);
        return high === 0 ? 0 : (high - low) / high;
      };
      const normalLineHeightRatio = 1.2;
      const typography = [];
      const spacing = [];
      const grid = [];
      const color = [];
      const saturationValues = [];
      for (const el of Array.from(document.querySelectorAll('body *'))) {
        if (!visible(el)) continue;
        const style = window.getComputedStyle(el);
        const box = el.getBoundingClientRect();
        if (hasOwnText(el)) {
          const fontSize = parseFloat(style.fontSize);
          const lineHeight = parseFloat(style.lineHeight);
          if (Number.isFinite(fontSize) && fontSize > 0) typography.push(fontSize);
          if (Number.isFinite(lineHeight) && lineHeight > 0 && Number.isFinite(fontSize) && fontSize > 0) {
            spacing.push(lineHeight / fontSize);
          } else if (style.lineHeight.trim().toLowerCase() === 'normal' && Number.isFinite(fontSize) && fontSize > 0) {
            spacing.push(normalLineHeightRatio);
          }
        }
        if (box.width > 0) grid.push(Number(((box.left / 8) % 12).toFixed(3)));
        for (const value of [style.color, style.backgroundColor]) {
          const rgb = parseRgb(value);
          if (!rgb) continue;
          color.push(rgbString(rgb));
          saturationValues.push(saturation(rgb));
        }
      }
      return { typography, spacing, grid, color, saturation: saturationValues };
    }
    """


def parse_viewport(value: str) -> dict[str, int]:
    """Parse the benchmark's portable WIDTHxHEIGHT viewport notation."""
    try:
        width, height = value.lower().split("x", 1)
        parsed = {"width": int(width), "height": int(height)}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid viewport {value!r}") from exc
    if parsed["width"] <= 0 or parsed["height"] <= 0:
        raise ValueError(f"invalid viewport {value!r}")
    return parsed


def feature_object(raw: Mapping[str, Any]) -> PageFeatures:
    """Validate a computed-style result before it enters the scorer."""
    values: dict[str, list[Any]] = {}
    for dimension in FEATURE_NAMES:
        samples = raw.get(dimension)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"browser feature extractor returned no {dimension} samples")
        values[dimension] = samples
    try:
        return PageFeatures(
            typography=[float(value) for value in values["typography"]],
            spacing=[float(value) for value in values["spacing"]],
            grid=[float(value) for value in values["grid"]],
            color=[str(value).lower() for value in values["color"]],
            saturation=[float(value) for value in values["saturation"]],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("browser feature extractor returned non-numeric feature values") from exc


def safe_artifact_path(root: Path, value: Any) -> Path:
    """Resolve a manifest artifact without permitting path traversal."""
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or "\\" in str(value):
        raise ValueError("artifact path must be portable and relative")
    resolved = root / path
    if not resolved.is_file():
        raise ValueError(f"artifact is missing: {path}")
    return resolved


def _chain_label(chain: Sequence[OperatorName]) -> str:
    return ">".join(chain) if chain else "fallback"


def browser_beam_search(
    html: str,
    scorer: BiasScorer,
    evaluate_features: Callable[[str], PageFeatures],
    *,
    config: BeamSearchConfig,
) -> dict[str, Any]:
    """Run the bounded controller while scoring every candidate in Chromium."""
    initial_features = evaluate_features(html)
    initial_score = scorer.score(initial_features).total
    initial_exposure = static_low_contrast_exposure(html) if config.enforce_contrast_guardrail else 0
    best = {
        "chain": tuple(),
        "html": html,
        "features": initial_features,
        "score": initial_score,
    }
    frontier = [best]
    evaluated_candidates = 0
    rejected_by_contrast = 0
    for _depth in range(1, config.max_depth + 1):
        candidates: list[dict[str, Any]] = []
        for node in frontier:
            chain = tuple(node["chain"])
            for operator in config.operators:
                if not config.allow_operator_reuse and operator in chain:
                    continue
                candidate_chain = chain + (operator,)
                candidate_html = apply_operator(str(node["html"]), operator)
                if config.enforce_contrast_guardrail:
                    exposure = static_low_contrast_exposure(candidate_html)
                    if exposure > initial_exposure + config.contrast_exposure_tolerance:
                        rejected_by_contrast += 1
                        continue
                candidate_features = evaluate_features(candidate_html)
                evaluated_candidates += 1
                candidates.append({
                    "chain": candidate_chain,
                    "html": candidate_html,
                    "features": candidate_features,
                    "score": scorer.score(candidate_features).total,
                })
        if not candidates:
            break
        candidates.sort(key=lambda item: (float(item["score"]), tuple(item["chain"])))
        frontier = candidates[: max(1, config.beam_width)]
        if float(frontier[0]["score"]) < float(best["score"]):
            best = frontier[0]
    return {
        "initial_html": html,
        "initial_features": initial_features,
        "initial_score": initial_score,
        "selected_html": best["html"],
        "selected_features": best["features"],
        "selected_score": best["score"],
        "selected_operator_chain": list(best["chain"]),
        "selected_operator_label": _chain_label(best["chain"]),
        "delta": float(best["score"]) - initial_score,
        "evaluated_candidate_n": evaluated_candidates,
        "contrast_rejected_candidate_n": rejected_by_contrast,
    }


def _blocked_network_route(route: Any) -> None:
    """Keep the browser-scored replay offline; archived HTML must be self-contained."""
    route.abort()


def _artifact_name(record_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id).strip("._") or "record"
    return f"{normalized}__{hashlib.sha256(record_id.encode()).hexdigest()[:12]}.html"


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta"]) for row in rows]
    return {
        "n_groups": len(rows),
        "changed_n": sum(delta < 0 for delta in deltas),
        "fallback_n": sum(delta == 0 for delta in deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "operator_chain_counts": {
            label: sum(row["selected_operator_label"] == label for row in rows)
            for label in sorted({str(row["selected_operator_label"]) for row in rows})
        },
    }


def evaluate_browser_scored_alignment(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path,
    output_artifact_root: Path,
    chrome: Path | None = None,
    timeout_ms: int = 30_000,
    config: BeamSearchConfig | None = None,
    max_records: int | None = None,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Run browser-scored alignment on the manifest's held-out AI test records."""
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when supplied")
    records = validate_manifest(manifest)
    train = [record for record in records if record.source == "human" and record.split == "train"]
    test_ai = [record for record in records if record.source == "ai" and record.split == "test"]
    if not test_ai:
        raise ValueError("manifest requires AI test records for browser-scored alignment")
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("manifest records must be a list")
    raw_by_id = {str(record.get("id")): record for record in raw_records if isinstance(record, Mapping)}
    if len(raw_by_id) != len(raw_records):
        raise ValueError("manifest record IDs must be unique mappings")
    references = fit_reference_by_page_type(train)
    scorers = {
        page_type: BiasScorer(reference, penalty="l2", nonparametric_dims=("typography",))
        for page_type, reference in references.items()
    }
    active_config = config or BeamSearchConfig()
    output_artifact_root.mkdir(parents=True, exist_ok=True)
    corrected_root = output_artifact_root / "corrected_html"
    corrected_root.mkdir(exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - optional browser dependency
        raise RuntimeError('Install browser dependencies with: pip install -e ".[browser]"') from exc

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    viewport = parse_viewport(str(manifest["metadata"].get("scoring_viewport", "390x844")))
    selected_test_ai = sorted(test_ai, key=lambda item: item.id)
    if max_records is not None:
        selected_test_ai = selected_test_ai[:max_records]
    with sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {"headless": True}
        if chrome is not None and chrome.exists():
            launch_kwargs["executable_path"] = str(chrome)
        browser = playwright.chromium.launch(**launch_kwargs)
        try:
            browser_version = browser.version
            for index, record in enumerate(selected_test_ai, start=1):
                context = browser.new_context(viewport=viewport)
                context.route("**/*", _blocked_network_route)
                page = context.new_page()
                try:
                    raw = raw_by_id[record.id]
                    provenance = raw.get("provenance")
                    if not isinstance(provenance, Mapping):
                        raise ValueError(f"AI record {record.id!r} is missing provenance")
                    artifacts = provenance.get("artifacts")
                    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("rendered_html"), Mapping):
                        raise ValueError(f"AI record {record.id!r} is missing rendered_html provenance")
                    html_path = safe_artifact_path(artifact_root, artifacts["rendered_html"].get("path"))
                    expected_digest = str(artifacts["rendered_html"].get("sha256", ""))
                    if sha256_file(html_path) != expected_digest:
                        raise ValueError(f"AI record {record.id!r} rendered HTML digest mismatch")
                    initial_html = html_path.read_text(encoding="utf-8")

                    def evaluate_features(candidate_html: str, browser_page: Any = page) -> PageFeatures:
                        browser_page.set_content(candidate_html, wait_until="load", timeout=timeout_ms)
                        browser_page.evaluate("document.fonts.ready")
                        browser_page.wait_for_timeout(40)
                        return feature_object(browser_page.evaluate(computed_style_feature_script()))

                    result = browser_beam_search(
                        initial_html,
                        scorers[record.page_type],
                        evaluate_features,
                        config=active_config,
                    )
                    corrected_path = corrected_root / _artifact_name(record.id)
                    corrected_path.write_text(str(result["selected_html"]), encoding="utf-8")
                    rows.append({
                        "id": record.id,
                        "group_id": record.group_id,
                        "page_type": record.page_type,
                        "model_id": record.model_id,
                        "initial_score": result["initial_score"],
                        "selected_score": result["selected_score"],
                        "delta": result["delta"],
                        "selected_operator_chain": result["selected_operator_chain"],
                        "selected_operator_label": result["selected_operator_label"],
                        "evaluated_candidate_n": result["evaluated_candidate_n"],
                        "contrast_rejected_candidate_n": result["contrast_rejected_candidate_n"],
                        "initial_rendered_html_sha256": sha256_file(html_path),
                        "selected_html_artifact": {
                            "path": corrected_path.relative_to(output_artifact_root).as_posix(),
                            "sha256": sha256_file(corrected_path),
                        },
                    })
                    print(f"[{index}/{len(selected_test_ai)}] {record.id}: {result['delta']:.6f}")
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    failures.append({"id": record.id, "error": str(exc)})
                    print(f"[{index}/{len(selected_test_ai)}] {record.id}: failed: {exc}")
                finally:
                    context.close()
        finally:
            browser.close()
    by_model = {
        model: _summary([row for row in rows if row["model_id"] == model])
        for model in sorted({str(row["model_id"]) for row in rows})
    }
    by_page_type = {
        page_type: _summary([row for row in rows if row["page_type"] == page_type])
        for page_type in sorted({str(row["page_type"]) for row in rows})
    }
    return {
        "schema": ALIGNMENT_SCHEMA,
        "benchmark": manifest["metadata"]["benchmark_name"],
        "version": manifest["metadata"]["version"],
        "reference_fit_split": "human_train_only",
        "evaluation_split": "held_out_ai_test",
        "n_available_test_ai_groups": len(test_ai),
        "n_evaluated_test_ai_groups": len(rows),
        "status": "complete" if not failures else "partial_with_failure_ledger",
        "failures": failures,
        "browser": {"engine": "chromium", "version": browser_version, "viewport": manifest["metadata"].get("scoring_viewport", "390x844")},
        "controller": {
            "beam_width": active_config.beam_width,
            "max_depth": active_config.max_depth,
            "operators": list(active_config.operators),
            "allow_operator_reuse": active_config.allow_operator_reuse,
            "enforce_contrast_guardrail": active_config.enforce_contrast_guardrail,
            "contrast_exposure_tolerance": active_config.contrast_exposure_tolerance,
            "scoring": "computed_style_reference_l2_w1",
        },
        "summary": _summary(rows),
        "by_model": by_model,
        "by_page_type": by_page_type,
        "records": rows,
        "interpretation": (
            "The reference is fit only from the frozen historical-reference train split and every AI page is in the "
            "held-out test split. Because the controller selects a lower value of this same reference-fit objective "
            "when one exists, score direction is a controller-consistency result, not independent evidence of visual "
            "quality, human authorship, preference, accessibility, or generalization beyond this release."
        ),
    }


def audit_browser_scored_alignment(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    input_artifact_root: Path,
    output_artifact_root: Path,
) -> dict[str, Any]:
    """Fail closed on incomplete or altered browser-scored alignment artifacts."""
    failures: list[dict[str, str]] = []
    try:
        records = validate_manifest(manifest)
        expected_ai = {
            record.id: record
            for record in records
            if record.source == "ai" and record.split == "test"
        }
        raw_records = manifest.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("manifest records must be a list")
        raw_by_id = {str(record.get("id")): record for record in raw_records if isinstance(record, Mapping)}
        if len(raw_by_id) != len(raw_records):
            raise ValueError("manifest records must contain unique mappings")
        if result.get("schema") != ALIGNMENT_SCHEMA:
            raise ValueError(f"result schema must be {ALIGNMENT_SCHEMA!r}")
        if result.get("status") != "complete" or result.get("failures"):
            raise ValueError("alignment result must be complete with no failure ledger entries")
        output_rows = result.get("records")
        if not isinstance(output_rows, list):
            raise ValueError("alignment result records must be a list")
        observed_ids = [str(row.get("id")) for row in output_rows if isinstance(row, Mapping)]
        if len(observed_ids) != len(output_rows) or set(observed_ids) != set(expected_ai):
            raise ValueError("alignment result IDs do not exactly match held-out AI test IDs")
        for row in output_rows:
            assert isinstance(row, Mapping)
            record_id = str(row["id"])
            try:
                raw = raw_by_id[record_id]
                provenance = raw.get("provenance")
                if not isinstance(provenance, Mapping):
                    raise ValueError("missing input provenance")
                artifacts = provenance.get("artifacts")
                if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("rendered_html"), Mapping):
                    raise ValueError("missing input rendered_html artifact")
                initial_path = safe_artifact_path(input_artifact_root, artifacts["rendered_html"].get("path"))
                expected_initial_digest = str(artifacts["rendered_html"].get("sha256", ""))
                if sha256_file(initial_path) != expected_initial_digest:
                    raise ValueError("input rendered_html digest mismatch")
                if row.get("initial_rendered_html_sha256") != expected_initial_digest:
                    raise ValueError("result initial rendered_html digest mismatch")
                selected = row.get("selected_html_artifact")
                if not isinstance(selected, Mapping):
                    raise ValueError("missing selected HTML artifact")
                selected_path = safe_artifact_path(output_artifact_root, selected.get("path"))
                if sha256_file(selected_path) != selected.get("sha256"):
                    raise ValueError("selected HTML artifact digest mismatch")
                initial_score = float(row["initial_score"])
                selected_score = float(row["selected_score"])
                delta = float(row["delta"])
                if not all(math.isfinite(value) for value in (initial_score, selected_score, delta)):
                    raise ValueError("non-finite alignment score")
                if not math.isclose(selected_score - initial_score, delta, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("selected score minus initial score does not equal delta")
                if delta > 1e-9:
                    raise ValueError("selected controller delta is positive")
            except (OSError, TypeError, ValueError) as exc:
                failures.append({"id": record_id, "error": str(exc)})
    except (TypeError, ValueError) as exc:
        failures.append({"id": "<result>", "error": str(exc)})
    return {
        "schema": "webmark_browser_scored_alignment_audit_v1",
        "n_expected_ai_test_groups": len(expected_ai) if "expected_ai" in locals() else 0,
        "n_checked": len(expected_ai) - len(failures) if "expected_ai" in locals() else 0,
        "n_failed": len(failures),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "note": (
            "This audit checks held-out ID coverage, input and selected-output hashes, and score arithmetic. "
            "It does not establish quality, preference, accessibility, or authorship."
        ),
    }
