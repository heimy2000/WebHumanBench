#!/usr/bin/env python3
"""Build minimal v1 AI source-run and exclusion ledgers from assembly artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.release import sha256_file


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_ledgers(
    source_run: dict[str, Any],
    ai_archive: dict[str, Any],
    manifest: dict[str, Any],
    *,
    source_run_sha256: str,
    ai_archive_sha256: str,
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_rows = source_run.get("per_page")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source run requires non-empty per_page rows")
    records = [
        {
            "id": str(row["id"]),
            "index": int(row["index"]),
            "model_id": str(row["model"]),
            "page_type": str(row["page_type"]),
            "scenario": str(row["scenario"]),
            "seed": int(row["seed"]),
            "status": str(row["status"]),
        }
        for row in raw_rows
    ]
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("source run contains duplicate page ids")

    archive_exclusions = ai_archive.get("exclusions")
    if not isinstance(archive_exclusions, list):
        raise ValueError("AI archive requires an exclusions list")
    planned = {row["id"]: row for row in records}
    exclusions = []
    for row in archive_exclusions:
        page_id = str(row["page_id"])
        if page_id not in planned:
            raise ValueError(f"excluded page is absent from source run: {page_id}")
        error = str(row.get("error", ""))
        if "requested resources outside its archived HTML" not in error:
            raise ValueError(f"excluded page is not an offline-closure failure: {page_id}")
        source = planned[page_id]
        exclusions.append({
            "error": error,
            "model_id": source["model_id"],
            "page_id": page_id,
            "page_type": source["page_type"],
            "reason_code": "external_network_dependency",
        })

    manifest_ai = [
        record for record in manifest.get("records", []) if record.get("source") == "ai"
    ]
    retained_ids = {
        str(record.get("provenance", {}).get("prompt_id")) for record in manifest_ai
    }
    excluded_ids = {row["page_id"] for row in exclusions}
    if "None" in retained_ids:
        raise ValueError("manifest AI row is missing provenance.prompt_id")
    if retained_ids.intersection(excluded_ids):
        raise ValueError("a page is both retained and excluded")
    if retained_ids.union(excluded_ids) != set(planned):
        raise ValueError("retained and excluded pages do not close the source run")

    config = source_run.get("config", {})
    source_ledger = {
        "schema": "webhumanbench_v1_ai_source_run_ledger_v1",
        "source_run_sha256": source_run_sha256,
        "planned_pages": len(records),
        "generation_protocol": {
            "max_tokens": config.get("max_tokens"),
            "models": config.get("models"),
            "page_types": config.get("page_types"),
            "pages_per_type": config.get("pages_per_type"),
            "seed": config.get("seed"),
            "temperature": config.get("temperature"),
        },
        "records": sorted(records, key=lambda row: row["id"]),
        "note": (
            "This ledger records the planned 600-page generation run only. It omits the "
            "archived controller outputs that shared the original execution file."
        ),
    }
    exclusion_ledger = {
        "schema": "webhumanbench_v1_ai_exclusion_ledger_v1",
        "source_run_sha256": source_run_sha256,
        "source_archive_sha256": ai_archive_sha256,
        "benchmark_manifest_sha256": manifest_sha256,
        "retained_pages": len(retained_ids),
        "excluded_pages": len(excluded_ids),
        "selection_rule": (
            "offline_local_render_with_complete_feature_arrays_and_no_external_network_dependency"
        ),
        "selection_uses_reference_fit_score": False,
        "exclusions": sorted(exclusions, key=lambda row: row["page_id"]),
        "note": (
            "Every omitted page failed the fixed offline-closure rule before reference-fit scoring."
        ),
    }
    return source_ledger, exclusion_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--ai-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-ledger-output", type=Path, required=True)
    parser.add_argument("--exclusion-ledger-output", type=Path, required=True)
    args = parser.parse_args()

    source_ledger, exclusion_ledger = build_ledgers(
        _load(args.source_run),
        _load(args.ai_archive),
        _load(args.manifest),
        source_run_sha256=sha256_file(args.source_run),
        ai_archive_sha256=sha256_file(args.ai_archive),
        manifest_sha256=sha256_file(args.manifest),
    )
    _write(args.source_ledger_output, source_ledger)
    _write(args.exclusion_ledger_output, exclusion_ledger)
    print(f"planned pages: {source_ledger['planned_pages']}")
    print(f"retained/excluded: {exclusion_ledger['retained_pages']}/{exclusion_ledger['excluded_pages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
