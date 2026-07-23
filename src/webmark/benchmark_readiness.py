"""Release gates that distinguish a WebHumanBench pilot from a formal corpus.

Candidate repositories, source audits, and partial captures never count toward
the gate.  Only records in a validated, group-disjoint benchmark manifest are
counted.  The formal profile implements the published 1,200 historical-source,
six-page-type expansion target: 120/40/40 human train/dev/test groups per type.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .human_likeness import BenchmarkRecord, validate_manifest
from .page_type import PAGE_TYPES

READINESS_SCHEMA = "webmark_benchmark_readiness_audit_v1"
FORMAL_PROFILE_NAME = "webhumanbench_formal_v1"
FORMAL_PROFILE = {
    "page_types": tuple(PAGE_TYPES),
    "min_human_train_groups_per_type": 120,
    "min_human_dev_groups_per_type": 40,
    "min_human_test_groups_per_type": 40,
    "min_ai_test_groups_per_type": 100,
}


def _group_ids(
    records: list[BenchmarkRecord], *, source: str, split: str, page_type: str
) -> set[str]:
    return {
        record.group_id
        for record in records
        if record.source == source and record.split == split and record.page_type == page_type
    }


def audit_benchmark_readiness(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed formal-benchmark readiness audit for a manifest."""
    records = validate_manifest(manifest)
    required_page_types = set(FORMAL_PROFILE["page_types"])
    declared_page_types = set(manifest["metadata"].get("required_page_types", []))
    per_page_type: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, Any]] = []
    for page_type in FORMAL_PROFILE["page_types"]:
        observed = {
            "human_train_groups": len(_group_ids(records, source="human", split="train", page_type=page_type)),
            "human_dev_groups": len(_group_ids(records, source="human", split="dev", page_type=page_type)),
            "human_test_groups": len(_group_ids(records, source="human", split="test", page_type=page_type)),
            "ai_test_groups": len(_group_ids(records, source="ai", split="test", page_type=page_type)),
        }
        required = {
            "human_train_groups": FORMAL_PROFILE["min_human_train_groups_per_type"],
            "human_dev_groups": FORMAL_PROFILE["min_human_dev_groups_per_type"],
            "human_test_groups": FORMAL_PROFILE["min_human_test_groups_per_type"],
            "ai_test_groups": FORMAL_PROFILE["min_ai_test_groups_per_type"],
        }
        deficits = {name: required[name] - observed[name] for name in required if observed[name] < required[name]}
        per_page_type[page_type] = {"observed": observed, "required": required, "deficits": deficits}
        if deficits:
            findings.append({"page_type": page_type, "deficits": deficits})
    if declared_page_types != required_page_types:
        findings.insert(
            0,
            {
                "page_type": "manifest_metadata",
                "deficits": {
                    "required_page_types": {
                        "expected": sorted(required_page_types),
                        "observed": sorted(declared_page_types),
                    }
                },
            },
        )
    status = "formal_benchmark_ready" if not findings else "pilot_only"
    return {
        "schema": READINESS_SCHEMA,
        "profile": FORMAL_PROFILE_NAME,
        "benchmark": manifest["metadata"]["benchmark_name"],
        "version": manifest["metadata"]["version"],
        "status": status,
        "formal_historical_source_target": 1_200,
        "formal_ai_test_group_target": 600,
        "per_page_type": per_page_type,
        "findings": findings,
        "counting_policy": (
            "Only validated scoring groups in this manifest count. Discovery queues, pending source audits, "
            "partial captures, and failed captures are excluded."
        ),
        "interpretation": (
            "This is a corpus-coverage gate, not a proof of individual human authorship, aesthetics, "
            "accessibility, or legal redistributability."
        ),
    }
