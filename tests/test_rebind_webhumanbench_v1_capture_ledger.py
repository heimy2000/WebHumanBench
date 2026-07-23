from __future__ import annotations

import importlib.util
from pathlib import Path

from webmark.release import canonical_json_sha256


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rebind_webhumanbench_v1_capture_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("rebind_webhumanbench_v1_capture_ledger", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(ids: list[str]) -> dict[str, object]:
    return {"sources": [{"id": source_id, "page_type": "fixture"} for source_id in ids]}


def _ledger(manifest: dict[str, object]) -> dict[str, object]:
    source_digest = canonical_json_sha256(manifest)
    source_receipts = {
        "source_manifest_sha256": source_digest,
        "records": [{"source_id": "keep"}, {"source_id": "exclude"}],
    }
    vendor_receipts = {
        "source_manifest_sha256": source_digest,
        "records": [{"source_id": "keep"}, {"source_id": "exclude"}],
    }
    return {
        "source_manifest_sha256": source_digest,
        "source_receipts": source_receipts,
        "source_receipts_sha256": canonical_json_sha256(source_receipts),
        "vendor_receipts": vendor_receipts,
        "vendor_receipts_sha256": canonical_json_sha256(vendor_receipts),
        "records": [
            {"source_id": "keep", "viewport": "390x844", "payload": "unchanged"},
            {"source_id": "keep", "viewport": "430x932", "payload": "unchanged"},
            {"source_id": "exclude", "viewport": "390x844"},
            {"source_id": "exclude", "viewport": "430x932"},
        ],
    }


def test_rebind_filters_only_explicitly_excluded_source_records() -> None:
    module = _load_module()
    input_manifest = _manifest(["keep", "exclude"])
    output_manifest = _manifest(["keep"])
    ledger = _ledger(input_manifest)

    rebound, receipt = module.rebind_capture_ledger(input_manifest, ledger, output_manifest)

    assert receipt["excluded_source_ids"] == ["exclude"]
    assert receipt["retained_capture_records"] == 2
    assert [row["source_id"] for row in rebound["source_receipts"]["records"]] == ["keep"]
    assert [row["source_id"] for row in rebound["vendor_receipts"]["records"]] == ["keep"]
    assert rebound["records"] == ledger["records"][:2]
    assert rebound["source_receipts_sha256"] == canonical_json_sha256(rebound["source_receipts"])
    assert rebound["vendor_receipts_sha256"] == canonical_json_sha256(rebound["vendor_receipts"])
