from __future__ import annotations

import json
from pathlib import Path

import pytest

from webmark.release_bundle import (
    DOCUMENT_FILES,
    RESULT_DIRECTORIES,
    RESULT_FILES,
    RUNTIME_SCRIPT_FILES,
    referenced_capture_artifact_paths,
    referenced_closure_manifest_paths,
    runtime_script_dependency_closure,
    stage_webhumanbench_release,
    stage_webhumanbench_v02_release,
)


def _artifact(path: str) -> dict[str, str]:
    return {"path": path, "sha256": "a" * 64}


def _release_inputs() -> tuple[dict[str, object], dict[str, object]]:
    capture_ledger: dict[str, object] = {
        "records": [
            {
                "artifacts": {
                    "html": _artifact("html/source.html"),
                    "screenshot": _artifact("screenshots/source.png"),
                }
            }
        ],
        "vendor_receipts": {
            "records": [
                {
                    "overlay_entrypoint": _artifact("snapshot_overlays/source/overlay/index.html"),
                    "overlay_files": [_artifact("snapshot_overlays/source/overlay/index.html")],
                    "vendor_assets": [
                        {"artifact": _artifact("snapshot_overlays/source/vendor/font.ttf")}
                    ],
                }
            ]
        },
    }
    ai_records: dict[str, object] = {
        "records": [
            {"provenance": {"artifacts": {"screenshot": _artifact("ai/screenshots/ai.png")}}}
        ]
    }
    return capture_ledger, ai_records


def test_referenced_capture_artifact_paths_are_deduplicated_and_sorted() -> None:
    capture_ledger, ai_records = _release_inputs()

    paths = referenced_capture_artifact_paths(capture_ledger, ai_records)

    assert [path.as_posix() for path in paths] == [
        "ai/screenshots/ai.png",
        "html/source.html",
        "screenshots/source.png",
        "snapshot_overlays/source/overlay/index.html",
        "snapshot_overlays/source/vendor/font.ttf",
    ]


def test_referenced_closure_manifests_are_deduplicated_and_sorted() -> None:
    source_receipts = {
        "records": [
            {"closure_manifest_path": "z.json"},
            {"closure_manifest_path": "nested/a.json"},
            {"closure_manifest_path": "z.json"},
        ]
    }

    paths = referenced_closure_manifest_paths(source_receipts)

    assert [path.as_posix() for path in paths] == ["nested/a.json", "z.json"]


def test_prompt_factorial_runtime_dependencies_are_staged_transitively() -> None:
    paths = runtime_script_dependency_closure(("scripts/run_prompt_factorial.py",))

    assert [path.as_posix() for path in paths] == [
        "scripts/capture_open_mobile_reference.py",
        "scripts/run_external_600_siliconflow.py",
        "scripts/run_prompt_factorial.py",
    ]


def test_stage_release_excludes_unreferenced_capture_files(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    source_data = code_root / "benchmark" / "webhumanbench_v0_2"
    capture_ledger, ai_records = _release_inputs()
    benchmark_manifest = {
        "metadata": {
            "benchmark_name": "WebHumanBench",
            "release_revision": "r" * 64,
            "version": "test",
        }
    }
    source_manifest = {"metadata": {"release_revision": "r" * 64}}
    files = {
        "source_manifest.json": source_manifest,
        "capture_ledger.json": capture_ledger,
        "split_assignments.json": {},
        "ai_records.json": ai_records,
        "webhumanbench_manifest.json": benchmark_manifest,
        "promotion_receipt.json": {},
    }
    for name, payload in files.items():
        path = source_data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    for relative_path in referenced_capture_artifact_paths(capture_ledger, ai_records):
        path = source_data / "captures" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.as_posix().encode("utf-8"))
    unreferenced = source_data / "captures" / "ai" / "unreferenced.html"
    unreferenced.parent.mkdir(parents=True, exist_ok=True)
    unreferenced.write_text("not part of the benchmark", encoding="utf-8")
    (source_data / "captures" / "._metadata").write_text("metadata", encoding="utf-8")

    for name in ("CITATION.cff", "LICENSE", "README.md", "pyproject.toml"):
        path = code_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    for relative in (*DOCUMENT_FILES, *RUNTIME_SCRIPT_FILES):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    package_init = code_root / "src" / "webmark" / "__init__.py"
    package_init.parent.mkdir(parents=True, exist_ok=True)
    package_init.write_text("", encoding="utf-8")
    for name in RESULT_FILES:
        path = code_root / "results" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for name in RESULT_DIRECTORIES:
        path = code_root / "results" / name / "corrected_html" / "fixture.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<main />", encoding="utf-8")

    stage_root = tmp_path / "stage"
    result = stage_webhumanbench_v02_release(code_root, stage_root)

    assert result["referenced_capture_artifact_count"] == 5
    assert (stage_root / "RELEASE_MANIFEST.json").is_file()
    assert not (
        stage_root
        / "code"
        / "benchmark"
        / "webhumanbench_v0_2"
        / "captures"
        / "ai"
        / "unreferenced.html"
    ).exists()
    assert not any(path.name.startswith("._") for path in stage_root.rglob("*"))
    release_manifest = json.loads(
        (stage_root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert release_manifest["payload_tree_sha256"] == result["payload_tree_sha256"]
    cache = stage_root / "code" / "src" / "webmark" / "__pycache__" / "release_bundle.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    resumed = stage_webhumanbench_v02_release(code_root, stage_root, resume=True)
    assert resumed["payload_tree_sha256"] == result["payload_tree_sha256"]
    assert not cache.parent.exists()
    refreshed = stage_webhumanbench_v02_release(code_root, stage_root, resume=True, refresh=True)
    assert refreshed["payload_tree_sha256"] == result["payload_tree_sha256"]


def test_generic_stage_rejects_a_nested_release_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one benchmark directory"):
        stage_webhumanbench_release(
            tmp_path,
            tmp_path / "stage",
            release_directory="webhumanbench/v1",
            result_files=(),
            document_files=(),
            runtime_script_files=(),
        )


def test_generic_stage_accepts_a_version_specific_core_receipt(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    source_data = code_root / "benchmark" / "webhumanbench_v1"
    capture_ledger, ai_records = _release_inputs()
    benchmark_manifest = {
        "metadata": {
            "benchmark_name": "WebHumanBench",
            "release_revision": "r" * 64,
            "version": "1.0.0",
        }
    }
    source_manifest = {"metadata": {"release_revision": "r" * 64}}
    files = {
        "source_manifest.json": source_manifest,
        "capture_ledger.json": capture_ledger,
        "split_assignments.json": {},
        "ai_records.json": ai_records,
        "webhumanbench_manifest.json": benchmark_manifest,
        "corpus_assembly_receipt.json": {"schema": "fixture"},
    }
    for name, payload in files.items():
        path = source_data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for relative_path in referenced_capture_artifact_paths(capture_ledger, ai_records):
        path = source_data / "captures" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.as_posix().encode("utf-8"))
    for name in ("CITATION.cff", "LICENSE", "README.md", "pyproject.toml"):
        path = code_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    package_init = code_root / "src" / "webmark" / "__init__.py"
    package_init.parent.mkdir(parents=True, exist_ok=True)
    package_init.write_text("", encoding="utf-8")
    release_test = code_root / "tests" / "test_release_fixture.py"
    release_test.parent.mkdir(parents=True)
    release_test.write_text("def test_release_fixture():\n    assert True\n", encoding="utf-8")

    stage_root = tmp_path / "stage"
    stage_webhumanbench_release(
        code_root,
        stage_root,
        release_directory="webhumanbench_v1",
        result_files=(),
        document_files=(),
        runtime_script_files=(),
        test_files=("tests/test_release_fixture.py",),
        core_benchmark_files=(
            "source_manifest.json",
            "capture_ledger.json",
            "split_assignments.json",
            "ai_records.json",
            "webhumanbench_manifest.json",
            "corpus_assembly_receipt.json",
        ),
    )

    staged = stage_root / "code" / "benchmark" / "webhumanbench_v1"
    assert (staged / "corpus_assembly_receipt.json").is_file()
    assert not (staged / "promotion_receipt.json").exists()
    assert (stage_root / "code" / "tests" / "test_release_fixture.py").is_file()


def test_generic_stage_copies_only_referenced_closure_manifests(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    source_data = code_root / "benchmark" / "webhumanbench_v1"
    capture_ledger, ai_records = _release_inputs()
    capture_ledger["source_receipts"] = {"records": [{"closure_manifest_path": "keep.json"}]}
    benchmark_manifest = {
        "metadata": {
            "benchmark_name": "WebHumanBench",
            "release_revision": "r" * 64,
            "version": "1.0.0",
        }
    }
    source_manifest = {"metadata": {"release_revision": "r" * 64}}
    for name, payload in {
        "source_manifest.json": source_manifest,
        "capture_ledger.json": capture_ledger,
        "ai_records.json": ai_records,
        "webhumanbench_manifest.json": benchmark_manifest,
    }.items():
        path = source_data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for relative_path in referenced_capture_artifact_paths(capture_ledger, ai_records):
        path = source_data / "captures" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.as_posix().encode("utf-8"))
    (source_data / "closure_manifests").mkdir(parents=True)
    (source_data / "closure_manifests" / "keep.json").write_text("keep", encoding="utf-8")
    (source_data / "closure_manifests" / "exclude.json").write_text("exclude", encoding="utf-8")
    for name in ("CITATION.cff", "LICENSE", "README.md", "pyproject.toml"):
        path = code_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    (code_root / "src" / "webmark").mkdir(parents=True)
    (code_root / "src" / "webmark" / "__init__.py").write_text("", encoding="utf-8")

    stage_root = tmp_path / "stage"
    stage_webhumanbench_release(
        code_root,
        stage_root,
        release_directory="webhumanbench_v1",
        result_files=(),
        document_files=(),
        runtime_script_files=(),
        core_benchmark_files=(
            "source_manifest.json",
            "capture_ledger.json",
            "ai_records.json",
            "webhumanbench_manifest.json",
        ),
        closure_manifest_directory="closure_manifests",
    )

    staged = stage_root / "code" / "benchmark" / "webhumanbench_v1" / "closure_manifests"
    assert (staged / "keep.json").is_file()
    assert not (staged / "exclude.json").exists()


def test_analysis_only_stage_retains_ledgers_without_raw_capture_artifacts(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    source_data = code_root / "benchmark" / "webhumanbench_v1"
    capture_ledger, ai_records = _release_inputs()
    benchmark_manifest = {
        "metadata": {
            "benchmark_name": "WebHumanBench",
            "release_revision": "r" * 64,
            "version": "1.0.0",
        }
    }
    for name, payload in {
        "source_manifest.json": {"metadata": {"release_revision": "r" * 64}},
        "capture_ledger.json": capture_ledger,
        "ai_records.json": ai_records,
        "webhumanbench_manifest.json": benchmark_manifest,
    }.items():
        path = source_data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for relative_path in referenced_capture_artifact_paths(capture_ledger, ai_records):
        path = source_data / "captures" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raw artifact", encoding="utf-8")
    for name in ("CITATION.cff", "LICENSE", "README.md", "pyproject.toml"):
        path = code_root / name
        path.write_text(name, encoding="utf-8")
    (code_root / "src" / "webmark").mkdir(parents=True)
    (code_root / "src" / "webmark" / "__init__.py").write_text("", encoding="utf-8")

    manifest = stage_webhumanbench_release(
        code_root,
        tmp_path / "stage",
        release_directory="webhumanbench_v1",
        result_files=(),
        document_files=(),
        runtime_script_files=(),
        core_benchmark_files=(
            "source_manifest.json",
            "capture_ledger.json",
            "ai_records.json",
            "webhumanbench_manifest.json",
        ),
        include_capture_artifacts=False,
    )

    staged_data = tmp_path / "stage" / "code" / "benchmark" / "webhumanbench_v1"
    assert (staged_data / "capture_ledger.json").is_file()
    assert not (staged_data / "captures").exists()
    assert not (tmp_path / "stage" / "code" / "RELEASE.md").exists()
    assert manifest["distribution_profile"] == "analysis-only"
    assert manifest["referenced_capture_artifact_count"] == 5
    assert manifest["included_capture_artifact_count"] == 0
