from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from webmark.release import canonical_json_sha256  # isort: skip


COMMIT = "a" * 40
PROTOCOL = {
    "browser_engine": "chromium",
    "browser_version": "150.0.7871.115",
    "locale": "en-US",
    "timezone": "UTC",
    "color_scheme": "light",
    "reduced_motion": "reduce",
    "device_scale_factor": 1,
}


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "assemble_webhumanbench_v1_corpus.py"
    spec = importlib.util.spec_from_file_location("assemble_webhumanbench_v1_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(source_id: str, page_type: str) -> dict[str, object]:
    repository = f"https://github.com/example/{source_id}"
    return {
        "id": source_id,
        "page_type": page_type,
        "group_id": f"source-{source_id}",
        "repository_url": repository,
        "repository_created_at": "2020-01-01T00:00:00Z",
        "commit_sha": COMMIT,
        "commit_authored_at": "2022-12-31T23:59:59Z",
        "license_spdx": "MIT",
        "license_url": f"{repository}/blob/{COMMIT}/LICENSE",
        "entrypoint": "index.html",
        "entrypoint_evidence_url": f"{repository}/blob/{COMMIT}/index.html",
        "build_command": "static_snapshot_with_vendored_assets",
        "capture_url": f"{repository}/blob/{COMMIT}/index.html",
        "capture_commit_evidence_url": f"{repository}/commit/{COMMIT}",
        "capture_method": "pinned_local_static_snapshot",
        "provenance_evidence": [
            {"kind": "pinned_git_history", "evidence_urls": [f"{repository}/commit/{COMMIT}"]},
            {"kind": "source_project_identity", "evidence_urls": [repository]},
        ],
        "viewports": ["390x844", "430x932"],
    }


def _manifest(
    sources: list[dict[str, object]], *, status: str, required_page_types: list[str]
) -> dict[str, object]:
    return {
        "schema": "webmark_open_mobile_reference_v2",
        "metadata": {
            "dataset_name": "fixture",
            "version": "fixture",
            "manifest_license": "CC-BY-4.0",
            "data_license": "CC-BY-4.0",
            "capture_browser": "150.0.7871.115",
            "capture_protocol": PROTOCOL,
            "temporal_cutoff": "2023-01-01T00:00:00Z",
            "temporal_policy": "before_cutoff",
            "primary_mobile_viewports": ["390x844", "430x932"],
            "required_page_types": required_page_types,
            "min_sources_per_page_type": 1,
            "provenance_policy": {
                "mode": "historical_open_source_evidence_v1",
                "min_distinct_evidence_kinds": 2,
            },
            "release_status": status,
            "release_revision": "b" * 64,
        },
        "sources": sources,
    }


def _screen(manifest: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    ledger = {
        "schema": "fixture_capture_ledger",
        "source_manifest_sha256": canonical_json_sha256(manifest),
    }
    screen = {
        "schema": "webmark_reference_render_integrity_screen_v1",
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "capture_ledger_sha256": canonical_json_sha256(ledger),
        "sources": [
            {"source_id": source["id"], "status": "pass"} for source in manifest["sources"]
        ],
    }
    return ledger, screen


def _inputs():
    module = _load_module()
    candidate_types = ["saas_landing", "docs_homepage", "developer_tool", "dashboard_shell"]
    candidate_sources = [
        _source(f"candidate-{page_type}-{index}", page_type)
        for page_type in candidate_types
        for index in range(4)
    ] + [_source("candidate-product", "product_showcase")]
    candidate = _manifest(
        candidate_sources,
        status="candidate",
        required_page_types=[*candidate_types, "product_showcase"],
    )
    candidate_ledger, candidate_screen = _screen(candidate)
    recovery = _manifest(
        [_source(f"recovery-product-{index}", "product_showcase") for index in range(3)],
        status="candidate",
        required_page_types=["product_showcase"],
    )
    recovery_ledger, recovery_screen = _screen(recovery)
    portfolio = _manifest(
        [_source(f"portfolio-{index}", "portfolio_showcase") for index in range(4)],
        status="public",
        required_page_types=["portfolio_showcase"],
    )
    return (
        module,
        candidate,
        candidate_ledger,
        candidate_screen,
        recovery,
        recovery_ledger,
        recovery_screen,
        portfolio,
    )


def test_assembly_creates_a_public_six_type_source_manifest() -> None:
    inputs = _inputs()
    manifest, receipt = inputs[0].assemble_webhumanbench_v1_corpus(*inputs[1:], version="1.0.0")

    assert manifest["metadata"]["release_status"] == "public"
    assert len(manifest["sources"]) == 24
    assert receipt["selected_source_counts"] == {
        "dashboard_shell": 4,
        "developer_tool": 4,
        "docs_homepage": 4,
        "portfolio_showcase": 4,
        "product_showcase": 4,
        "saas_landing": 4,
    }
    assert receipt["capture_plan"]["expected_human_capture_records"] == 48
    assert receipt["source_manifest_sha256"] == canonical_json_sha256(manifest)


def test_assembly_rejects_screen_with_a_stale_manifest_digest() -> None:
    inputs = list(_inputs())
    screen = copy.deepcopy(inputs[3])
    screen["source_manifest_sha256"] = "0" * 64
    inputs[3] = screen

    with pytest.raises(ValueError, match="does not bind its source manifest"):
        inputs[0].assemble_webhumanbench_v1_corpus(*inputs[1:], version="1.0.0")


def test_assembly_records_an_explicit_exclusion_when_coverage_remains_valid() -> None:
    inputs = list(_inputs())
    candidate = copy.deepcopy(inputs[1])
    candidate["sources"].append(_source("candidate-developer-tool-extra", "developer_tool"))
    candidate_ledger, candidate_screen = _screen(candidate)
    inputs[1] = candidate
    inputs[2] = candidate_ledger
    inputs[3] = candidate_screen

    manifest, receipt = inputs[0].assemble_webhumanbench_v1_corpus(
        *inputs[1:],
        version="1.0.0",
        excluded_source_ids=["candidate-developer-tool-extra"],
        exclusion_reason="fixture preflight exclusion",
    )

    assert len(manifest["sources"]) == 24
    assert receipt["selected_source_counts"]["developer_tool"] == 4
    assert receipt["explicit_public_exclusions"] == [
        {
            "source_id": "candidate-developer-tool-extra",
            "reason": "fixture preflight exclusion",
        }
    ]


def test_assembly_rejects_an_exclusion_that_breaks_page_type_coverage() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="at least 4 sources per page type"):
        inputs[0].assemble_webhumanbench_v1_corpus(
            *inputs[1:],
            version="1.0.0",
            excluded_source_ids=["candidate-developer_tool-0"],
            exclusion_reason="fixture preflight exclusion",
        )
