#!/usr/bin/env python3
"""Assemble the released WebHumanBench v1 human-reference source corpus.

The release deliberately admits only sources with fixed provenance inputs:
passed candidate/recovery render screens and the already public v0.2 portfolio
cohort.  It does not relabel candidate rows as public by fiat.  The output
receipt binds each input manifest, capture ledger, render screen, selected
source ID, and final immutable release revision.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.open_reference import (  # noqa: E402
    HISTORICAL_EVIDENCE_SCHEMA,
    PRIMARY_MOBILE_VIEWPORTS,
    validate_open_reference_manifest,
)
from webmark.release import canonical_json_sha256  # noqa: E402

ASSEMBLY_SCHEMA = "webhumanbench_v1_corpus_assembly_receipt_v1"
RENDER_SCREEN_SCHEMA = "webmark_reference_render_integrity_screen_v1"
REQUIRED_PAGE_TYPES = (
    "saas_landing",
    "docs_homepage",
    "product_showcase",
    "developer_tool",
    "dashboard_shell",
    "portfolio_showcase",
)
PRIMARY_CANDIDATE_PAGE_TYPES = frozenset(REQUIRED_PAGE_TYPES).difference({"portfolio_showcase"})
MIN_SOURCES_PER_PAGE_TYPE = 4


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metadata(manifest: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    return _mapping(manifest.get("metadata"), f"{context} metadata")


def _validate_render_screen(
    screen: Mapping[str, Any],
    manifest: Mapping[str, Any],
    capture_ledger: Mapping[str, Any],
    *,
    context: str,
) -> set[str]:
    if screen.get("schema") != RENDER_SCREEN_SCHEMA:
        raise ValueError(f"{context} render screen must use {RENDER_SCREEN_SCHEMA!r}")
    if screen.get("source_manifest_sha256") != canonical_json_sha256(manifest):
        raise ValueError(f"{context} render screen does not bind its source manifest")
    if screen.get("capture_ledger_sha256") != canonical_json_sha256(capture_ledger):
        raise ValueError(f"{context} render screen does not bind its capture ledger")
    rows = screen.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context} render screen requires non-empty sources")
    known_ids = {source.id for source in validate_open_reference_manifest(manifest)}
    passed: set[str] = set()
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{context} render screen sources[{index}]")
        source_id = str(row.get("source_id", "")).strip()
        if source_id not in known_ids:
            raise ValueError(f"{context} render screen references unknown source {source_id!r}")
        if source_id in seen:
            raise ValueError(f"{context} render screen has duplicate source {source_id!r}")
        seen.add(source_id)
        if row.get("status") == "pass":
            passed.add(source_id)
    if seen != known_ids:
        missing = sorted(known_ids.difference(seen))
        raise ValueError(f"{context} render screen omits source IDs: {', '.join(missing[:3])}")
    return passed


def _require_candidate_manifest(manifest: Mapping[str, Any], context: str) -> None:
    if manifest.get("schema") != HISTORICAL_EVIDENCE_SCHEMA:
        raise ValueError(f"{context} manifest must use historical-evidence schema")
    if _metadata(manifest, context).get("release_status") != "candidate":
        raise ValueError(f"{context} manifest must be marked release_status 'candidate'")


def _require_public_portfolio_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != HISTORICAL_EVIDENCE_SCHEMA:
        raise ValueError("portfolio manifest must use historical-evidence schema")
    metadata = _metadata(manifest, "portfolio manifest")
    if metadata.get("release_status") != "public":
        raise ValueError("portfolio manifest must be a public prior release")
    sources = validate_open_reference_manifest(manifest)
    if not sources or any(source.page_type != "portfolio_showcase" for source in sources):
        raise ValueError("portfolio manifest may contain only portfolio_showcase sources")


def _protocol(manifests: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str]:
    first_metadata = _metadata(manifests[0], "source manifest")
    protocol = _mapping(first_metadata.get("capture_protocol"), "source manifest capture_protocol")
    browser = str(first_metadata.get("capture_browser", "")).strip()
    if not browser:
        raise ValueError("source manifest metadata requires capture_browser")
    for index, manifest in enumerate(manifests[1:], start=1):
        metadata = _metadata(manifest, f"source manifest {index}")
        if metadata.get("capture_protocol") != protocol:
            raise ValueError("all source inputs must use the same capture protocol")
        if metadata.get("capture_browser") != browser:
            raise ValueError("all source inputs must use the same capture browser")
    return copy.deepcopy(dict(protocol)), browser


def _selected_source_rows(
    manifest: Mapping[str, Any],
    passed_ids: set[str],
    *,
    allowed_page_types: frozenset[str],
    input_name: str,
    input_manifest_sha256: str,
    capture_ledger_sha256: str | None,
    render_screen_sha256: str | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw_source in manifest.get("sources", []):
        source = _mapping(raw_source, f"{input_name} source")
        source_id = str(source.get("id", ""))
        page_type = str(source.get("page_type", ""))
        if source_id not in passed_ids or page_type not in allowed_page_types:
            continue
        row = copy.deepcopy(dict(source))
        evidence: dict[str, str] = {
            "input_name": input_name,
            "input_source_manifest_sha256": input_manifest_sha256,
        }
        if capture_ledger_sha256 is not None:
            evidence["input_capture_ledger_sha256"] = capture_ledger_sha256
        if render_screen_sha256 is not None:
            evidence["input_render_screen_sha256"] = render_screen_sha256
        row["v1_selection_evidence"] = evidence
        selected.append(row)
    return selected


def _validate_selected_sources(sources: Sequence[dict[str, Any]]) -> Counter[str]:
    ids = [str(source["id"]) for source in sources]
    groups = [str(source["group_id"]) for source in sources]
    repositories = [str(source["repository_url"]).rstrip("/").lower() for source in sources]
    for label, values in (
        ("source IDs", ids),
        ("source groups", groups),
        ("repositories", repositories),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"assembled corpus contains duplicate {label}")
    counts = Counter(str(source["page_type"]) for source in sources)
    missing = [
        page_type
        for page_type in REQUIRED_PAGE_TYPES
        if counts.get(page_type, 0) < MIN_SOURCES_PER_PAGE_TYPE
    ]
    if missing:
        details = ", ".join(f"{page_type}={counts.get(page_type, 0)}" for page_type in missing)
        raise ValueError(
            f"assembled corpus requires at least {MIN_SOURCES_PER_PAGE_TYPE} sources per page type: {details}"
        )
    return counts


def _normalize_exclusions(
    excluded_source_ids: Sequence[str], exclusion_reason: str | None
) -> tuple[tuple[str, ...], str | None]:
    """Validate explicit publication exclusions before source selection."""
    normalized = tuple(sorted({str(source_id).strip() for source_id in excluded_source_ids}))
    if any(not source_id for source_id in normalized):
        raise ValueError("excluded source IDs must be non-empty")
    reason = str(exclusion_reason or "").strip() or None
    if normalized and reason is None:
        raise ValueError("explicit source exclusions require an exclusion_reason")
    if not normalized and reason is not None:
        raise ValueError("exclusion_reason requires at least one excluded source ID")
    return normalized, reason


def assemble_webhumanbench_v1_corpus(
    candidate_manifest: Mapping[str, Any],
    candidate_capture_ledger: Mapping[str, Any],
    candidate_render_screen: Mapping[str, Any],
    product_recovery_manifest: Mapping[str, Any],
    product_recovery_capture_ledger: Mapping[str, Any],
    product_recovery_render_screen: Mapping[str, Any],
    portfolio_manifest: Mapping[str, Any],
    *,
    version: str,
    excluded_source_ids: Sequence[str] = (),
    exclusion_reason: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the public v1 source manifest and its deterministic assembly receipt."""
    if not version.strip():
        raise ValueError("version must be non-empty")
    excluded_ids, normalized_exclusion_reason = _normalize_exclusions(
        excluded_source_ids, exclusion_reason
    )
    _require_candidate_manifest(candidate_manifest, "candidate")
    _require_candidate_manifest(product_recovery_manifest, "product recovery")
    _require_public_portfolio_manifest(portfolio_manifest)
    validate_open_reference_manifest(candidate_manifest)
    product_sources = validate_open_reference_manifest(product_recovery_manifest)
    portfolio_sources = validate_open_reference_manifest(portfolio_manifest)
    if any(source.page_type != "product_showcase" for source in product_sources):
        raise ValueError("product recovery manifest may contain only product_showcase sources")

    candidate_passed = _validate_render_screen(
        candidate_render_screen,
        candidate_manifest,
        candidate_capture_ledger,
        context="candidate",
    )
    product_passed = _validate_render_screen(
        product_recovery_render_screen,
        product_recovery_manifest,
        product_recovery_capture_ledger,
        context="product recovery",
    )
    protocol, browser = _protocol(
        (candidate_manifest, product_recovery_manifest, portfolio_manifest)
    )

    candidate_manifest_sha = canonical_json_sha256(candidate_manifest)
    candidate_ledger_sha = canonical_json_sha256(candidate_capture_ledger)
    candidate_screen_sha = canonical_json_sha256(candidate_render_screen)
    recovery_manifest_sha = canonical_json_sha256(product_recovery_manifest)
    recovery_ledger_sha = canonical_json_sha256(product_recovery_capture_ledger)
    recovery_screen_sha = canonical_json_sha256(product_recovery_render_screen)
    portfolio_manifest_sha = canonical_json_sha256(portfolio_manifest)

    selected = _selected_source_rows(
        candidate_manifest,
        candidate_passed,
        allowed_page_types=PRIMARY_CANDIDATE_PAGE_TYPES,
        input_name="candidate_render_screen_r1",
        input_manifest_sha256=candidate_manifest_sha,
        capture_ledger_sha256=candidate_ledger_sha,
        render_screen_sha256=candidate_screen_sha,
    )
    selected.extend(
        _selected_source_rows(
            product_recovery_manifest,
            product_passed,
            allowed_page_types=frozenset({"product_showcase"}),
            input_name="product_recovery_render_screen_r3",
            input_manifest_sha256=recovery_manifest_sha,
            capture_ledger_sha256=recovery_ledger_sha,
            render_screen_sha256=recovery_screen_sha,
        )
    )
    selected.extend(
        _selected_source_rows(
            portfolio_manifest,
            {source.id for source in portfolio_sources},
            allowed_page_types=frozenset({"portfolio_showcase"}),
            input_name="webhumanbench_v0_2_public_portfolio_release",
            input_manifest_sha256=portfolio_manifest_sha,
            capture_ledger_sha256=None,
            render_screen_sha256=None,
        )
    )
    selectable_ids = {str(source["id"]) for source in selected}
    unknown_exclusions = sorted(set(excluded_ids).difference(selectable_ids))
    if unknown_exclusions:
        raise ValueError(
            "explicit source exclusions must identify selected sources: "
            + ", ".join(unknown_exclusions[:3])
        )
    selected = [source for source in selected if str(source["id"]) not in excluded_ids]
    selected.sort(
        key=lambda source: (REQUIRED_PAGE_TYPES.index(str(source["page_type"])), str(source["id"]))
    )
    counts = _validate_selected_sources(selected)

    selection_inputs = {
        "candidate_source_manifest_sha256": candidate_manifest_sha,
        "candidate_capture_ledger_sha256": candidate_ledger_sha,
        "candidate_render_screen_sha256": candidate_screen_sha,
        "product_recovery_source_manifest_sha256": recovery_manifest_sha,
        "product_recovery_capture_ledger_sha256": recovery_ledger_sha,
        "product_recovery_render_screen_sha256": recovery_screen_sha,
        "portfolio_source_manifest_sha256": portfolio_manifest_sha,
    }
    selection_policy = {
        "name": "screened_static_v1_with_public_portfolio_carryover",
        "candidate_page_types": sorted(PRIMARY_CANDIDATE_PAGE_TYPES),
        "candidate_admission": "dual-mobile render-integrity pass",
        "product_recovery_admission": "dual-mobile render-integrity pass",
        "portfolio_admission": "all sources from the verified v0.2 public portfolio release",
        "excluded_candidate_page_type": "portfolio_showcase",
        "min_sources_per_page_type": MIN_SOURCES_PER_PAGE_TYPE,
        "explicit_public_exclusion_policy": (
            "A source may be excluded only by explicit ID with a recorded publication reason; "
            "the retained corpus must still satisfy every page-type coverage minimum."
        ),
    }
    explicit_exclusions = [
        {"source_id": source_id, "reason": normalized_exclusion_reason}
        for source_id in excluded_ids
    ]
    revision_material = {
        "schema": ASSEMBLY_SCHEMA,
        "dataset_name": "WebHumanBench",
        "version": version,
        "selection_inputs": selection_inputs,
        "selection_policy": selection_policy,
        "selected_source_ids": [str(source["id"]) for source in selected],
        "selected_source_counts": dict(sorted(counts.items())),
        "explicit_public_exclusions": explicit_exclusions,
    }
    release_revision = canonical_json_sha256(revision_material)
    metadata = {
        "dataset_name": "WebHumanBench",
        "version": version,
        "manifest_license": "CC-BY-4.0",
        "data_license": (
            "Benchmark metadata and derived feature arrays: CC-BY-4.0; captured source and "
            "vendored artifacts: upstream terms."
        ),
        "capture_browser": browser,
        "capture_protocol": protocol,
        "temporal_cutoff": "2023-01-01T00:00:00Z",
        "temporal_policy": "before_cutoff",
        "primary_mobile_viewports": list(PRIMARY_MOBILE_VIEWPORTS),
        "required_page_types": list(REQUIRED_PAGE_TYPES),
        "min_sources_per_page_type": MIN_SOURCES_PER_PAGE_TYPE,
        "provenance_policy": {
            "mode": "historical_open_source_evidence_v1",
            "min_distinct_evidence_kinds": 2,
            "label_scope": "historical_open_source_temporal_provenance_proxy",
        },
        "release_status": "public",
        "release_revision": release_revision,
        "corpus_assembly": {
            "schema": ASSEMBLY_SCHEMA,
            "selection_policy": selection_policy["name"],
            "selection_inputs_sha256": canonical_json_sha256(selection_inputs),
        },
        "scope_note": (
            "v1 is a released, versioned historical open-source reference corpus. Its operational "
            "provenance label is not a verified claim about an individual creator or a human-preference label."
        ),
    }
    manifest = {"schema": HISTORICAL_EVIDENCE_SCHEMA, "metadata": metadata, "sources": selected}
    # Re-run the public schema validation after adding the release metadata.
    validate_open_reference_manifest(manifest)
    receipt = {
        "schema": ASSEMBLY_SCHEMA,
        "dataset_name": "WebHumanBench",
        "version": version,
        "release_revision": release_revision,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "selection_inputs": selection_inputs,
        "selection_policy": selection_policy,
        "selected_source_ids": [str(source["id"]) for source in selected],
        "selected_source_counts": dict(sorted(counts.items())),
        "explicit_public_exclusions": explicit_exclusions,
        "capture_plan": {
            "method": "pinned_local_static_snapshot",
            "viewports": list(PRIMARY_MOBILE_VIEWPORTS),
            "expected_human_capture_records": len(selected) * len(PRIMARY_MOBILE_VIEWPORTS),
        },
        "note": (
            "The final v1 capture ledger and benchmark manifest are generated after this source assembly. "
            "The receipt records only source admission and does not replace final artifact verification."
        ),
    }
    return manifest, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-capture-ledger", type=Path, required=True)
    parser.add_argument("--candidate-render-screen", type=Path, required=True)
    parser.add_argument("--product-recovery-manifest", type=Path, required=True)
    parser.add_argument("--product-recovery-capture-ledger", type=Path, required=True)
    parser.add_argument("--product-recovery-render-screen", type=Path, required=True)
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        default=[],
        help="selected source ID to exclude from this public release; requires --exclusion-reason",
    )
    parser.add_argument(
        "--exclusion-reason",
        help="publication reason recorded for every --exclude-source-id",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    manifest, receipt = assemble_webhumanbench_v1_corpus(
        _load_object(args.candidate_manifest),
        _load_object(args.candidate_capture_ledger),
        _load_object(args.candidate_render_screen),
        _load_object(args.product_recovery_manifest),
        _load_object(args.product_recovery_capture_ledger),
        _load_object(args.product_recovery_render_screen),
        _load_object(args.portfolio_manifest),
        version=args.version,
        excluded_source_ids=args.exclude_source_id,
        exclusion_reason=args.exclusion_reason,
    )
    for path, payload in ((args.output_manifest, manifest), (args.output_receipt, receipt)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "assembled WebHumanBench v1 source corpus: "
        f"{len(manifest['sources'])} sources, revision {manifest['metadata']['release_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
