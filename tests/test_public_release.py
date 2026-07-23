"""Tests for the public-release cross-artifact integrity contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.open_reference import HISTORICAL_EVIDENCE_SCHEMA  # isort: skip  # noqa: E402
from webmark.pinned_build import BUILD_RECEIPT_SCHEMA, output_tree_manifest  # isort: skip  # noqa: E402
from webmark.release import canonical_json_sha256, sha256_file, validate_public_release  # isort: skip  # noqa: E402


COMMIT = "a" * 40
REVISION = "b" * 40
PROTOCOL = {
    "browser_engine": "chromium",
    "browser_version": "142.0.0",
    "locale": "en-US",
    "timezone": "UTC",
    "color_scheme": "light",
    "reduced_motion": "reduce",
    "device_scale_factor": 1,
}


def _features(offset: float = 0.0) -> dict[str, object]:
    return {
        "typography": [16.0 + offset, 18.0 + offset, 20.0 + offset],
        "spacing": [1.4 + offset / 20, 1.6 + offset / 20],
        "grid": [4.0 + offset / 5, 5.0 + offset / 5],
        "color": ["rgb(31, 41, 55)", "rgb(248, 250, 252)", "rgb(15, 118, 110)"],
        "saturation": [0.35 + offset / 100, 0.45 + offset / 100],
    }


def _source(source_id: str, page_type: str) -> dict[str, object]:
    return {
        "id": source_id,
        "page_type": page_type,
        "group_id": f"site-{source_id}",
        "repository_url": f"https://github.com/example/{source_id}",
        "repository_created_at": "2020-01-01T00:00:00Z",
        "commit_sha": COMMIT,
        "commit_authored_at": "2022-12-31T23:59:59Z",
        "license_spdx": "MIT",
        "license_url": f"https://github.com/example/{source_id}/blob/{COMMIT}/LICENSE",
        "entrypoint": "apps/site/src/pages/index.astro",
        "entrypoint_evidence_url": f"https://github.com/example/{source_id}/blob/{COMMIT}/apps/site/src/pages/index.astro",
        "build_command": "npm run build",
        "capture_url": f"https://example.github.io/{source_id}/",
        "capture_commit_evidence_url": f"https://example.github.io/{source_id}/deployments/{COMMIT}",
        "capture_method": "pinned_local_static_checkout",
        "curation_reviewer_id": "curator-01",
        "curation_reviewed_at": "2026-01-01T00:00:00Z",
        "human_provenance_urls": [f"https://example.org/{source_id}/provenance"],
        "provenance_reviews": [
            {
                "reviewer_id": "curator-01",
                "reviewed_at": "2026-01-01T00:00:00Z",
                "decision": "admit",
                "evidence_urls": [f"https://example.org/{source_id}/review-01"],
            },
            {
                "reviewer_id": "curator-02",
                "reviewed_at": "2026-01-02T00:00:00Z",
                "decision": "admit",
                "evidence_urls": [f"https://example.org/{source_id}/review-02"],
            },
        ],
        "viewports": ["390x844", "430x932"],
    }


def _source_manifest() -> dict[str, object]:
    return {
        "schema": "webmark_open_mobile_reference_v1",
        "metadata": {
            "dataset_name": "Public fixture",
            "version": "1.0.0",
            "manifest_license": "CC-BY-4.0",
            "data_license": "CC-BY-4.0",
            "capture_browser": "Chromium 142",
            "temporal_cutoff": "2023-01-01T00:00:00Z",
            "primary_mobile_viewports": ["390x844", "430x932"],
            "required_page_types": ["saas_landing", "docs_homepage"],
            "min_sources_per_page_type": 1,
            "release_status": "public",
            "release_revision": REVISION,
            "capture_protocol": PROTOCOL,
            "curation_policy": {"min_independent_reviews": 2},
        },
        "sources": [_source("saas", "saas_landing"), _source("docs", "docs_homepage")],
    }


def _capture_record(tmp_path: Path, source_id: str, page_type: str, viewport: str, offset: float) -> dict[str, object]:
    stem = f"{source_id}__{viewport.replace('x', '_')}"
    html_path = tmp_path / "html" / f"{stem}.html"
    screenshot_path = tmp_path / "screenshots" / f"{stem}.png"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(f"<!doctype html><title>{stem}</title>", encoding="utf-8")
    screenshot_path.write_bytes(f"png:{stem}".encode("ascii"))
    features = _features(offset)
    return {
        "id": f"{source_id}@{viewport}",
        "source_id": source_id,
        "source": "human",
        "group_id": f"site-{source_id}",
        "page_type": page_type,
        "viewport": viewport,
        "capture_url": f"https://example.github.io/{source_id}/",
        "final_url": f"http://127.0.0.1:8000/{source_id}/",
        "http_status": 200,
        "capture_origin": {
            "mode": "pinned_local_static_checkout",
            "commit_sha": COMMIT,
            "checkout_tree_sha256": "c" * 64,
            "entrypoint_sha256": "d" * 64,
        },
        "captured_at": "2026-01-03T00:00:00Z",
        "features": features,
        "feature_sha256": canonical_json_sha256(features),
        "capture_html_sha256": sha256_file(html_path),
        "artifacts": {
            "html": {"path": html_path.relative_to(tmp_path).as_posix(), "sha256": sha256_file(html_path)},
            "screenshot": {
                "path": screenshot_path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(screenshot_path),
            },
        },
    }


def _release_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sources = _source_manifest()
    records = [
        _capture_record(tmp_path, "saas", "saas_landing", "390x844", 0.1),
        _capture_record(tmp_path, "saas", "saas_landing", "430x932", 0.2),
        _capture_record(tmp_path, "docs", "docs_homepage", "390x844", -0.1),
        _capture_record(tmp_path, "docs", "docs_homepage", "430x932", -0.2),
    ]
    captures = {
        "schema": "webmark_open_mobile_capture_v2",
        "source_manifest_sha256": canonical_json_sha256(sources),
        "capture_protocol": PROTOCOL,
        "runtime": {
            "browser_engine": "chromium",
            "browser_version": "142.0.0",
            "playwright_version": "1.61.0",
        },
        "feature_extractor_version": "computed-style-v2",
        "feature_extractor_script_sha256": "f" * 64,
        "captured_at": "2026-01-03T00:00:00Z",
        "records": records,
    }
    by_id = {str(record["id"]): record for record in records}

    def human_record(record_id: str, split: str) -> dict[str, object]:
        capture = by_id[record_id]
        return {
            "id": f"human-{record_id}-{split}",
            "source": "human",
            "split": split,
            "group_id": capture["group_id"],
            "leakage_group_id": capture["group_id"],
            "page_type": capture["page_type"],
            "viewport": capture["viewport"],
            "features": capture["features"],
            "provenance": {
                "source_id": capture["source_id"],
                "capture_id": capture["id"],
                "feature_sha256": capture["feature_sha256"],
                "capture_html_sha256": capture["capture_html_sha256"],
            },
        }

    # Public checks need each split type but a group may not cross splits. The
    # fixture duplicates no source, so it uses a deliberately small two-type
    # release after creating source-specific records for each split below.
    public_sources = _source_manifest()
    public_sources["sources"] = [
        _source(f"{page}-{split}", page_type)
        for split in ("train-a", "train-b", "dev", "test")
        for page, page_type in (("saas", "saas_landing"), ("docs", "docs_homepage"))
    ]
    capture_records = [
        _capture_record(tmp_path, str(source["id"]), str(source["page_type"]), viewport, index / 10)
        for index, source in enumerate(public_sources["sources"])
        for viewport in ("390x844", "430x932")
    ]
    captures = {
        "schema": "webmark_open_mobile_capture_v2",
        "source_manifest_sha256": canonical_json_sha256(public_sources),
        "capture_protocol": PROTOCOL,
        "runtime": {
            "browser_engine": "chromium",
            "browser_version": "142.0.0",
            "playwright_version": "1.61.0",
        },
        "feature_extractor_version": "computed-style-v2",
        "feature_extractor_script_sha256": "f" * 64,
        "captured_at": "2026-01-03T00:00:00Z",
        "records": capture_records,
    }
    by_id = {str(record["id"]): record for record in capture_records}

    def public_human(source_id: str, split: str) -> dict[str, object]:
        capture = by_id[f"{source_id}@390x844"]
        return {
            "id": f"human-{source_id}",
            "source": "human",
            "split": split,
            "group_id": capture["group_id"],
            "leakage_group_id": capture["group_id"],
            "page_type": capture["page_type"],
            "viewport": "390x844",
            "features": capture["features"],
            "provenance": {
                "source_id": source_id,
                "capture_id": capture["id"],
                "feature_sha256": capture["feature_sha256"],
                "capture_html_sha256": capture["capture_html_sha256"],
            },
        }

    def ai_record(page_type: str, model_id: str, offset: float) -> dict[str, object]:
        features = _features(offset)
        artifact_values = {
            "prompt": f"prompt:{page_type}",
            "generation_config": "{\"temperature\":0.2}",
            "raw_response": f"raw-response:{page_type}",
            "generated_html": f"<!doctype html><title>ai-{page_type}</title>",
            "rendered_html": f"<!doctype html><title>rendered-ai-{page_type}</title>",
            "screenshot": f"png:ai-{page_type}",
            "computed_features": json.dumps(features, sort_keys=True),
        }
        artifacts: dict[str, dict[str, str]] = {}
        digests: dict[str, str] = {}
        for name, content in artifact_values.items():
            path = tmp_path / "ai" / f"{page_type}__{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            artifacts[name] = {"path": path.relative_to(tmp_path).as_posix(), "sha256": sha256_file(path)}
            digests[name] = sha256_file(path)
        return {
            "id": f"ai-{page_type}",
            "source": "ai",
            "split": "test",
            "group_id": f"prompt-{page_type}-{model_id}",
            "leakage_group_id": f"prompt-{page_type}",
            "page_type": page_type,
            "viewport": "390x844",
            "model_id": model_id,
            "features": features,
            "provenance": {
                "prompt_id": f"prompt-{page_type}",
                "provider": "fixture-provider",
                "model_id": model_id,
                "generated_at": "2026-01-04T00:00:00Z",
                "prompt_sha256": digests["prompt"],
                "generation_config_sha256": digests["generation_config"],
                "raw_response_sha256": digests["raw_response"],
                "generated_html_sha256": digests["generated_html"],
                "rendered_html_sha256": digests["rendered_html"],
                "screenshot_sha256": digests["screenshot"],
                "computed_feature_sha256": canonical_json_sha256(features),
                "feature_extractor_version": "computed-style-v2",
                "artifacts": artifacts,
            },
        }

    benchmark = {
        "schema": "webmark_human_likeness_benchmark_v1",
        "metadata": {
            "benchmark_name": "Public fixture",
            "version": "1.0.0",
            "reference_scope": "fixture only",
            "license": "CC-BY-4.0",
            "required_page_types": ["saas_landing", "docs_homepage"],
            "mobile_test_share_min": 0.70,
            "release_status": "public",
            "release_revision": REVISION,
            "source_manifest_sha256": canonical_json_sha256(public_sources),
            "capture_ledger_sha256": canonical_json_sha256(captures),
            "feature_extractor_version": "computed-style-v2",
            "scoring_unit": "source_group",
            "scoring_viewport": "390x844",
        },
        "records": [
            public_human("saas-train-a", "train"),
            public_human("saas-train-b", "train"),
            public_human("docs-train-a", "train"),
            public_human("docs-train-b", "train"),
            public_human("saas-dev", "dev"),
            public_human("docs-dev", "dev"),
            public_human("saas-test", "test"),
            public_human("docs-test", "test"),
            ai_record("saas_landing", "model-a", 7.0),
            ai_record("docs_homepage", "model-a", -7.0),
        ],
    }
    return public_sources, captures, benchmark


def _historical_evidence_release_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sources, captures, benchmark = _release_fixture(tmp_path)
    sources["schema"] = HISTORICAL_EVIDENCE_SCHEMA
    sources["metadata"].pop("curation_policy")
    sources["metadata"]["provenance_policy"] = {
        "mode": "historical_open_source_evidence_v1",
        "min_distinct_evidence_kinds": 2,
    }
    for source in sources["sources"]:
        source.pop("curation_reviewer_id")
        source.pop("curation_reviewed_at")
        source.pop("human_provenance_urls")
        source.pop("provenance_reviews")
        source["provenance_evidence"] = [
            {
                "kind": "pinned_git_history",
                "evidence_urls": [source["capture_commit_evidence_url"]],
            },
            {
                "kind": "source_project_identity",
                "evidence_urls": [source["entrypoint_evidence_url"]],
            },
        ]
    captures["source_manifest_sha256"] = canonical_json_sha256(sources)
    benchmark["metadata"]["source_manifest_sha256"] = canonical_json_sha256(sources)
    benchmark["metadata"]["capture_ledger_sha256"] = canonical_json_sha256(captures)
    return sources, captures, benchmark


def _build_backed_release_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Convert one ordinary public fixture source into a receipt-backed build source."""
    sources, captures, benchmark = _release_fixture(tmp_path)
    source_id = "saas-train-a"
    source = next(item for item in sources["sources"] if item["id"] == source_id)
    recipe = {
        "schema": "webmark_pinned_local_build_recipe_v1",
        "package_manager": "npm",
        "working_directory": ".",
        "package_manifest": "package.json",
        "lockfile": "package-lock.json",
        "build_script": "build",
        "output_directory": "dist",
        "output_entrypoint": "index.html",
        "node_version": "v22.22.0",
        "package_manager_version": "10.9.4",
        "install_timeout_s": 300,
        "build_timeout_s": 300,
    }
    source["capture_method"] = "pinned_local_build"
    source["build_recipe"] = recipe
    output = tmp_path / "builds" / source_id / "output"
    output.mkdir(parents=True)
    (output / "index.html").write_text("<main>built fixture</main>", encoding="utf-8")
    output_manifest = output_tree_manifest(output)
    output_manifest_path = tmp_path / "builds" / source_id / "output_manifest.json"
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    install_log = tmp_path / "builds" / source_id / "install.log"
    build_log = tmp_path / "builds" / source_id / "build.log"
    install_log.write_text("npm ci --ignore-scripts\n", encoding="utf-8")
    build_log.write_text("npm run build\n", encoding="utf-8")
    receipt = {
        "source_id": source_id,
        "commit_sha": COMMIT,
        "entrypoint": source["entrypoint"],
        "entrypoint_sha256": "d" * 64,
        "checkout_tree_sha256": "c" * 64,
        "source_materialization": {
            "materialization_method": "github_api_tarball",
            "checkout_tree_sha256": "c" * 64,
            "checkout_file_tree_sha256": "e" * 64,
            "github_tree_sha": "f" * 40,
            "source_archive_sha256": "a" * 64,
        },
        "build_recipe_sha256": canonical_json_sha256(recipe),
        "package_manager": "npm",
        "package_manager_version": "10.9.4",
        "node_version": "v22.22.0",
        "lockfile": "package-lock.json",
        "lockfile_sha256": "b" * 64,
        "package_manifest_sha256": "c" * 64,
        "output_directory": f"builds/{source_id}/output",
        "output_entrypoint": "index.html",
        "output_entrypoint_sha256": sha256_file(output / "index.html"),
        "output_tree_sha256": canonical_json_sha256(output_manifest),
        "sandbox": {
            "backend": "sandbox-exec",
            "install_lifecycle_scripts": False,
            "install_network": "enabled_for_locked_dependency_resolution",
            "build_network": "denied",
            "install_profile_sha256": "d" * 64,
            "build_profile_sha256": "e" * 64,
        },
        "artifacts": {
            "install_log": {"path": install_log.relative_to(tmp_path).as_posix(), "sha256": sha256_file(install_log)},
            "build_log": {"path": build_log.relative_to(tmp_path).as_posix(), "sha256": sha256_file(build_log)},
            "output_manifest": {
                "path": output_manifest_path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(output_manifest_path),
            },
        },
        "built_at": "2026-01-03T00:00:00Z",
    }
    build_receipts = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(sources),
        "created_at": "2026-01-03T00:00:00Z",
        "records": [receipt],
    }
    for capture in captures["records"]:
        if capture["source_id"] != source_id:
            continue
        capture["capture_origin"] = {
            "mode": "pinned_local_build",
            "commit_sha": COMMIT,
            "checkout_tree_sha256": receipt["checkout_tree_sha256"],
            "entrypoint_sha256": receipt["entrypoint_sha256"],
            "build_receipt_sha256": canonical_json_sha256(receipt),
            "build_recipe_sha256": receipt["build_recipe_sha256"],
            "output_tree_sha256": receipt["output_tree_sha256"],
            "output_entrypoint_sha256": receipt["output_entrypoint_sha256"],
        }
    captures["source_manifest_sha256"] = canonical_json_sha256(sources)
    captures["build_receipts"] = build_receipts
    captures["build_receipts_sha256"] = canonical_json_sha256(build_receipts)
    benchmark["metadata"]["source_manifest_sha256"] = canonical_json_sha256(sources)
    benchmark["metadata"]["capture_ledger_sha256"] = canonical_json_sha256(captures)
    return sources, captures, benchmark


def test_public_release_cross_checks_json_and_artifact_hashes(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    result = validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)
    assert result["n_sources"] == 8
    assert result["n_captures"] == 16
    assert result["n_benchmark_records"] == 10
    assert result["artifact_hashes_checked"] is True


def test_public_release_accepts_explicit_historical_evidence_without_fake_reviewers(tmp_path: Path):
    sources, captures, benchmark = _historical_evidence_release_fixture(tmp_path)
    result = validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)
    assert result["provenance_policy_mode"] == "historical_open_source_evidence"


def test_public_release_audits_a_sandboxed_build_receipt_and_its_output(tmp_path: Path):
    sources, captures, benchmark = _build_backed_release_fixture(tmp_path)
    result = validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)
    assert result["n_sources"] == 8


def test_public_release_rejects_a_mutated_sandbox_build_output(tmp_path: Path):
    sources, captures, benchmark = _build_backed_release_fixture(tmp_path)
    (tmp_path / "builds" / "saas-train-a" / "output" / "index.html").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="output_tree_sha256"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_historical_evidence_release_rejects_single_evidence_kind(tmp_path: Path):
    sources, captures, benchmark = _historical_evidence_release_fixture(tmp_path)
    sources["sources"][0]["provenance_evidence"] = [
        sources["sources"][0]["provenance_evidence"][0]
    ]
    with pytest.raises(ValueError, match="historical provenance evidence"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_rejects_broken_capture_binding(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    benchmark["records"][0]["provenance"]["capture_id"] = "missing-capture"
    with pytest.raises(ValueError, match="unknown capture_id"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_rejects_mutated_artifact(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    first_html = captures["records"][0]["artifacts"]["html"]["path"]
    (tmp_path / str(first_html)).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_rejects_capture_from_a_different_pinned_commit(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    captures["records"][0]["capture_origin"]["commit_sha"] = "e" * 40
    with pytest.raises(ValueError, match="capture_origin.commit_sha"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_rejects_duplicate_scoring_capture_for_one_group(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    duplicate = dict(benchmark["records"][0])
    duplicate["id"] = "duplicate-human-saas-train"
    benchmark["records"].append(duplicate)
    with pytest.raises(ValueError, match="exactly one scoring record per group"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_rejects_missing_ai_archival_artifact(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    benchmark["records"][-1]["provenance"]["artifacts"].pop("raw_response")
    with pytest.raises(ValueError, match="raw_response"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_requires_each_admitted_source_to_be_scored_once(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    benchmark["records"] = [
        record
        for record in benchmark["records"]
        if record["id"] != "human-saas-test"
    ]
    with pytest.raises(ValueError, match="missing source records"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)


def test_public_release_requires_explicit_lineage_field(tmp_path: Path):
    sources, captures, benchmark = _release_fixture(tmp_path)
    benchmark["records"][0].pop("leakage_group_id")
    with pytest.raises(ValueError, match="explicit leakage_group_id"):
        validate_public_release(sources, captures, benchmark, artifact_root=tmp_path)
