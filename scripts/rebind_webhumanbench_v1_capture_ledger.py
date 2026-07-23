#!/usr/bin/env python3
"""Filter a complete capture ledger after an explicit public source exclusion.

The command never recaptures, edits, or sanitizes a retained artifact. It
validates the input ledger, retains only source IDs in the new manifest, and
rebinds aggregate manifest digests while preserving each retained source,
vendor, and capture record byte-for-byte.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.open_reference import validate_open_reference_manifest  # noqa: E402
from webmark.release import _validate_capture_ledger, canonical_json_sha256  # noqa: E402

REBIND_SCHEMA = "webhumanbench_capture_ledger_rebind_receipt_v1"


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _source_rows(manifest: Mapping[str, Any], context: str) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context} requires non-empty sources")
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{context} sources[{index}]")
        source_id = str(row.get("id", "")).strip()
        if not source_id or source_id in output:
            raise ValueError(f"{context} source IDs must be non-empty and unique")
        output[source_id] = row
    return output


def _filter_records(
    raw_rows: Any, *, retained_ids: set[str], context: str
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        raise ValueError(f"{context} requires a records list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, f"{context} records[{index}]")
        source_id = str(row.get("source_id", "")).strip()
        if not source_id or source_id in seen:
            raise ValueError(f"{context} records require unique non-empty source_id values")
        seen.add(source_id)
        if source_id in retained_ids:
            rows.append(copy.deepcopy(dict(row)))
    if {str(row["source_id"]) for row in rows} != retained_ids:
        missing = sorted(retained_ids.difference({str(row["source_id"]) for row in rows}))
        raise ValueError(f"{context} is missing retained source IDs: {', '.join(missing[:3])}")
    return rows


def _filter_capture_records(
    raw_rows: Any, *, retained_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        raise ValueError("capture ledger requires a records list")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, f"capture ledger records[{index}]")
        source_id = str(row.get("source_id", "")).strip()
        viewport = str(row.get("viewport", "")).strip()
        key = (source_id, viewport)
        if not source_id or not viewport or key in seen:
            raise ValueError("capture ledger records require unique source_id/viewport pairs")
        seen.add(key)
        if source_id in retained_ids:
            rows.append(copy.deepcopy(dict(row)))
    retained_capture_ids = {source_id for source_id, _ in seen if source_id in retained_ids}
    if retained_capture_ids != retained_ids:
        missing = sorted(retained_ids.difference(retained_capture_ids))
        raise ValueError(f"capture ledger is missing retained source IDs: {', '.join(missing[:3])}")
    return rows


def rebind_capture_ledger(
    input_source_manifest: Mapping[str, Any],
    input_capture_ledger: Mapping[str, Any],
    output_source_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a filtered ledger and a deterministic receipt for a strict source subset."""
    input_sources = _source_rows(input_source_manifest, "input source manifest")
    output_sources = _source_rows(output_source_manifest, "output source manifest")
    input_ids = set(input_sources)
    retained_ids = set(output_sources)
    if not retained_ids < input_ids:
        raise ValueError("output source manifest must be a strict subset of the input manifest")
    for source_id in sorted(retained_ids):
        if canonical_json_sha256(input_sources[source_id]) != canonical_json_sha256(
            output_sources[source_id]
        ):
            raise ValueError(f"retained source {source_id!r} differs from its input manifest row")

    input_source_sha = canonical_json_sha256(input_source_manifest)
    output_source_sha = canonical_json_sha256(output_source_manifest)
    if input_capture_ledger.get("source_manifest_sha256") != input_source_sha:
        raise ValueError("input capture ledger source_manifest_sha256 does not match input source manifest")
    source_receipts = _mapping(input_capture_ledger.get("source_receipts"), "input source receipts")
    vendor_receipts = _mapping(input_capture_ledger.get("vendor_receipts"), "input vendor receipts")
    if source_receipts.get("source_manifest_sha256") != input_source_sha:
        raise ValueError("input source receipts source_manifest_sha256 does not match input source manifest")
    if vendor_receipts.get("source_manifest_sha256") != input_source_sha:
        raise ValueError("input vendor receipts source_manifest_sha256 does not match input source manifest")
    if input_capture_ledger.get("source_receipts_sha256") != canonical_json_sha256(source_receipts):
        raise ValueError("input capture ledger source_receipts_sha256 does not match source receipts")
    if input_capture_ledger.get("vendor_receipts_sha256") != canonical_json_sha256(vendor_receipts):
        raise ValueError("input capture ledger vendor_receipts_sha256 does not match vendor receipts")

    rebound_source_receipts = copy.deepcopy(dict(source_receipts))
    rebound_source_receipts["source_manifest_sha256"] = output_source_sha
    rebound_source_receipts["records"] = _filter_records(
        source_receipts.get("records"), retained_ids=retained_ids, context="source receipts"
    )
    rebound_vendor_receipts = copy.deepcopy(dict(vendor_receipts))
    rebound_vendor_receipts["source_manifest_sha256"] = output_source_sha
    rebound_vendor_receipts["records"] = _filter_records(
        vendor_receipts.get("records"), retained_ids=retained_ids, context="vendor receipts"
    )

    rebound = copy.deepcopy(dict(input_capture_ledger))
    rebound["source_manifest_sha256"] = output_source_sha
    rebound["source_receipts"] = rebound_source_receipts
    rebound["source_receipts_sha256"] = canonical_json_sha256(rebound_source_receipts)
    rebound["vendor_receipts"] = rebound_vendor_receipts
    rebound["vendor_receipts_sha256"] = canonical_json_sha256(rebound_vendor_receipts)
    rebound["records"] = _filter_capture_records(
        input_capture_ledger.get("records"), retained_ids=retained_ids
    )
    rebound["rebind"] = {
        "schema": REBIND_SCHEMA,
        "input_source_manifest_sha256": input_source_sha,
        "input_capture_ledger_sha256": canonical_json_sha256(input_capture_ledger),
        "retained_source_ids": sorted(retained_ids),
        "excluded_source_ids": sorted(input_ids.difference(retained_ids)),
        "record_policy": (
            "Retained source-receipt, vendor-receipt, and capture-record payloads are copied "
            "unchanged; only aggregate source-manifest bindings and their hashes are updated."
        ),
    }
    receipt = {
        "schema": REBIND_SCHEMA,
        "input_source_manifest_sha256": input_source_sha,
        "input_capture_ledger_sha256": canonical_json_sha256(input_capture_ledger),
        "output_source_manifest_sha256": output_source_sha,
        "output_capture_ledger_sha256": canonical_json_sha256(rebound),
        "retained_source_ids": sorted(retained_ids),
        "excluded_source_ids": sorted(input_ids.difference(retained_ids)),
        "retained_capture_records": len(rebound["records"]),
        "policy": rebound["rebind"]["record_policy"],
    }
    return rebound, receipt


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_ledger(
    ledger: Mapping[str, Any], manifest: Mapping[str, Any], artifact_root: Path | None
) -> None:
    sources = validate_open_reference_manifest(manifest)
    metadata = _mapping(manifest.get("metadata"), "source manifest metadata")
    protocol = _mapping(metadata.get("capture_protocol"), "source manifest capture_protocol")
    _validate_capture_ledger(ledger, manifest, sources, protocol, artifact_root)
    if ledger.get("status") != "complete" or ledger.get("failures") != []:
        raise ValueError("capture ledger must be complete with no failures before rebinding")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-source-manifest", type=Path, required=True)
    parser.add_argument("--input-capture-ledger", type=Path, required=True)
    parser.add_argument("--output-source-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-capture-ledger", type=Path, required=True)
    parser.add_argument("--output-source-receipts", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    input_manifest = _load_object(args.input_source_manifest)
    input_ledger = _load_object(args.input_capture_ledger)
    output_manifest = _load_object(args.output_source_manifest)
    _validate_ledger(input_ledger, input_manifest, args.artifact_root)
    rebound, receipt = rebind_capture_ledger(input_manifest, input_ledger, output_manifest)
    _validate_ledger(rebound, output_manifest, args.artifact_root)
    for path, payload in (
        (args.output_capture_ledger, rebound),
        (args.output_source_receipts, rebound["source_receipts"]),
        (args.output_receipt, receipt),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "rebound WebHumanBench capture ledger: "
        f"{len(receipt['retained_source_ids'])} sources, {receipt['retained_capture_records']} captures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
