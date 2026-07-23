"""Tests for WebHumanBench protocol-boundary auditing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.benchmark_protocol_audit import build_protocol_audit
from webmark.release import sha256_file


def test_protocol_audit_binds_prompt_and_capture_interventions(tmp_path: Path) -> None:
    capture_root = tmp_path / "captures"
    prompt_path = capture_root / "ai" / "prompts" / "page-a.json"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        json.dumps([
            {
                "role": "system",
                "content": (
                    "You generate self-contained, production-like HTML/CSS webpages for a research benchmark."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Scenario/theme: test scenario\n"
                    "Page type label: saas_landing\n"
                    "No external assets, scripts, icon libraries, web fonts, iframes, or network calls.\n"
                    "Use literal CSS values.\n"
                    "The design should be plausible for a modern human-authored website.\n"
                    "Variation seed: 42"
                ),
            },
        ]),
        encoding="utf-8",
    )
    source_manifest = {
        "sources": [
            {
                "id": "source-a",
                "commit_authored_at": "2022-01-01T00:00:00Z",
            }
        ]
    }
    capture_ledger = {
        "captured_at": "2026-01-01T00:00:00Z",
        "vendor_receipts": {
            "records": [
                {
                    "source_id": "source-a",
                    "original_entrypoint_sha256": "a",
                    "snapshot_entrypoint_sha256": "b",
                    "vendor_assets": [
                        {"content_type": "text/css", "original_url": "https://example.invalid/a.css"}
                    ],
                    "removed_external_scripts": [],
                }
            ]
        },
    }
    benchmark_manifest = {
        "records": [
            {
                "source": "ai",
                "page_type": "saas_landing",
                "provenance": {
                    "prompt_id": "page-a",
                    "model_id": "model-a",
                    "generated_at": "2026-01-02T00:00:00Z",
                    "artifacts": {
                        "prompt": {
                            "path": "ai/prompts/page-a.json",
                            "sha256": sha256_file(prompt_path),
                        }
                    },
                },
            }
        ]
    }
    source_run_ledger = {
        "schema": "webhumanbench_v1_ai_source_run_ledger_v1",
        "records": [
            {
                "id": "page-a",
                "model_id": "model-a",
                "page_type": "saas_landing",
                "scenario": "test scenario",
                "seed": 42,
            }
        ],
    }

    result = build_protocol_audit(
        source_manifest,
        capture_ledger,
        benchmark_manifest,
        source_run_ledger,
        capture_root=capture_root,
    )

    assert result["reference_temporal_provenance"]["all_commits_before_cutoff"] is True
    assert result["reference_capture_interventions"]["vendored_asset_count"] == 1
    prompt = result["generated_prompt_protocol"]
    assert prompt["audited_retained_prompt_artifacts"] == 1
    assert all(item["fraction"] == 1 for item in prompt["required_clause_coverage"].values())

