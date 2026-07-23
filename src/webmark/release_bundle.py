"""Build a minimal, hash-indexed WebHumanBench release tree.

The development directory contains candidate snapshots and a larger model cache.
Those files are useful during curation but must not silently enter a benchmark
archive. This module copies only artifacts reachable from frozen release
ledgers and AI-record manifests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .release import canonical_json_sha256

RELEASE_BUNDLE_SCHEMA = "webhumanbench_release_bundle_v1"
V02_DIRECTORY = "webhumanbench_v0_2"
CORE_BENCHMARK_FILES = (
    "source_manifest.json",
    "capture_ledger.json",
    "split_assignments.json",
    "ai_records.json",
    "webhumanbench_manifest.json",
    "promotion_receipt.json",
)
RESULT_FILES = (
    "webhumanbench_v0_2_public_release_audit.json",
    "webhumanbench_v0_2_portfolio_baseline.json",
    "webhumanbench_v0_2_reference_design_profile.json",
    "webhumanbench_v0_2_reference_baselines_r1.json",
    "webhumanbench_v0_2_readiness_r1.json",
    "webhumanbench_v0_2_browser_alignment_r2.json",
    "webhumanbench_v0_2_browser_alignment_artifact_audit_r1.json",
)
RESULT_DIRECTORIES = ("webhumanbench_v0_2_browser_alignment_artifacts_r2",)
DOCUMENT_FILES = (
    "DATA_CARD.md",
    "THIRD_PARTY_ARTIFACT_NOTICES.md",
    "docs/FORMAL_BENCHMARK_GATE.md",
    "docs/HUMAN_LIKENESS_BENCHMARK.md",
    "docs/REPRODUCING.md",
    "docs/WEBHUMANBENCH_ALIGNMENT_EVALUATION.md",
    "docs/WEBHUMANBENCH_REFERENCE_DESIGN_PROFILE.md",
    "docs/WEBHUMANBENCH_V0_2_RELEASE.md",
    "benchmark/README.md",
)
RUNTIME_SCRIPT_FILES = (
    "scripts/analyze_webhumanbench_reference_profile.py",
    "scripts/audit_benchmark_readiness.py",
    "scripts/audit_public_release.py",
    "scripts/audit_webhumanbench_browser_alignment.py",
    "scripts/preflight_public_release.py",
    "scripts/run_human_likeness_benchmark.py",
    "scripts/run_reference_fit_baselines.py",
    "scripts/run_webhumanbench_browser_alignment.py",
    "scripts/stage_webhumanbench_release.py",
)
# Some release-facing entry points import local helper scripts rather than
# modules under ``src/webmark``. Keep their closure in the bundle contract so
# analysis-only reproduction does not depend on the development checkout.
RUNTIME_SCRIPT_DEPENDENCIES = {
    "scripts/run_prompt_factorial.py": (
        "scripts/capture_open_mobile_reference.py",
        "scripts/run_external_600_siliconflow.py",
    ),
}
ROOT_FILES = (
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "pyproject.toml",
)
OPTIONAL_ROOT_FILES = (".gitignore", "RELEASE.md")
IGNORED_NAMES = frozenset(
    {
        ".DS_Store",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "venv",
    }
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _relative_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires a non-empty path")
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in value:
        raise ValueError(f"{context} must be a portable relative path")
    return Path(*pure_path.parts)


def runtime_script_dependency_closure(runtime_script_files: Iterable[str]) -> tuple[Path, ...]:
    """Return declared release scripts plus their local runtime dependencies."""
    pending = [_relative_path(path, "runtime script") for path in runtime_script_files]
    resolved: set[Path] = set()
    while pending:
        script = pending.pop()
        if script in resolved:
            continue
        resolved.add(script)
        dependencies = RUNTIME_SCRIPT_DEPENDENCIES.get(script.as_posix(), ())
        pending.extend(_relative_path(path, "runtime script dependency") for path in dependencies)
    return tuple(sorted(resolved, key=lambda path: path.as_posix()))


def _artifact_path(value: Any, context: str) -> Path:
    artifact = _mapping(value, context)
    return _relative_path(artifact.get("path"), f"{context}.path")


def referenced_capture_artifact_paths(
    capture_ledger: Mapping[str, Any], ai_records: Mapping[str, Any]
) -> tuple[Path, ...]:
    """Return the exact capture-root paths reachable from frozen v0.2 indices."""
    paths: set[Path] = set()
    captures = capture_ledger.get("records")
    if not isinstance(captures, list) or not captures:
        raise ValueError("capture ledger requires non-empty records")
    for record_index, raw_record in enumerate(captures):
        record = _mapping(raw_record, f"capture ledger records[{record_index}]")
        artifacts = _mapping(
            record.get("artifacts"), f"capture ledger records[{record_index}].artifacts"
        )
        for artifact_name, artifact in artifacts.items():
            paths.add(_artifact_path(artifact, f"capture artifact {artifact_name!r}"))

    vendor_receipts = _mapping(
        capture_ledger.get("vendor_receipts"), "capture ledger vendor_receipts"
    )
    vendor_records = vendor_receipts.get("records")
    if not isinstance(vendor_records, list):
        raise ValueError("capture ledger vendor_receipts.records must be a list")
    for receipt_index, raw_receipt in enumerate(vendor_records):
        receipt = _mapping(raw_receipt, f"vendor receipt {receipt_index}")
        paths.add(
            _artifact_path(
                receipt.get("overlay_entrypoint"),
                f"vendor receipt {receipt_index}.overlay_entrypoint",
            )
        )
        overlays = receipt.get("overlay_files")
        if not isinstance(overlays, list):
            raise ValueError(f"vendor receipt {receipt_index}.overlay_files must be a list")
        for overlay_index, overlay in enumerate(overlays):
            paths.add(
                _artifact_path(
                    overlay, f"vendor receipt {receipt_index}.overlay_files[{overlay_index}]"
                )
            )
        vendor_assets = receipt.get("vendor_assets")
        if not isinstance(vendor_assets, list):
            raise ValueError(f"vendor receipt {receipt_index}.vendor_assets must be a list")
        for asset_index, raw_asset in enumerate(vendor_assets):
            asset = _mapping(
                raw_asset, f"vendor receipt {receipt_index}.vendor_assets[{asset_index}]"
            )
            paths.add(
                _artifact_path(
                    asset.get("artifact"),
                    f"vendor receipt {receipt_index}.vendor_assets[{asset_index}].artifact",
                )
            )

    records = ai_records.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("AI records require non-empty records")
    for record_index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"AI record {record_index}")
        provenance = _mapping(record.get("provenance"), f"AI record {record_index}.provenance")
        artifacts = _mapping(
            provenance.get("artifacts"), f"AI record {record_index}.provenance.artifacts"
        )
        for artifact_name, artifact in artifacts.items():
            paths.add(
                _artifact_path(artifact, f"AI record {record_index} artifact {artifact_name!r}")
            )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def referenced_closure_manifest_paths(source_receipts: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the source-closure manifests referenced by retained source receipts."""
    rows = source_receipts.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source receipts require non-empty records")
    paths: set[Path] = set()
    for receipt_index, raw_receipt in enumerate(rows):
        receipt = _mapping(raw_receipt, f"source receipt {receipt_index}")
        paths.add(
            _relative_path(
                receipt.get("closure_manifest_path"),
                f"source receipt {receipt_index}.closure_manifest_path",
            )
        )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path, *, replace_existing: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required release file is missing: {source}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"release destination exists but is not a file: {destination}")
        if source.stat().st_size == destination.stat().st_size and _sha256(source) == _sha256(
            destination
        ):
            return
        if not replace_existing:
            raise ValueError(f"release destination differs from source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_tree(source: Path, destination: Path, *, replace_existing: bool = False) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"required release directory is missing: {source}")
    for candidate in source.rglob("*"):
        relative = candidate.relative_to(source)
        if any(part in IGNORED_NAMES or part.startswith("._") for part in relative.parts):
            continue
        if candidate.is_symlink():
            raise ValueError(f"release source cannot contain symlinks: {candidate}")
        if candidate.is_file():
            _copy_file(candidate, destination / relative, replace_existing=replace_existing)


def _copy_paths(
    source_root: Path,
    destination_root: Path,
    paths: Iterable[Path],
    *,
    replace_existing: bool = False,
) -> None:
    for relative_path in paths:
        _copy_file(
            source_root / relative_path,
            destination_root / relative_path,
            replace_existing=replace_existing,
        )


def _remove_transient_staging_files(root: Path) -> None:
    """Remove local caches and operating-system metadata from a staged tree."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix(), reverse=True):
        if path.is_dir() and path.name in IGNORED_NAMES:
            shutil.rmtree(path)
        elif path.is_file() and (
            path.name.startswith("._")
            or path.name in {".DS_Store", ".coverage"}
            or path.suffix in {".pyc", ".pyo"}
        ):
            path.unlink()


def _payload_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "RELEASE_MANIFEST.json":
            continue
        if path.name.startswith("._") or path.name == ".DS_Store":
            raise ValueError(
                f"release tree unexpectedly contains operating-system metadata: {path}"
            )
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return entries


def _payload_tree_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def stage_webhumanbench_release(
    code_root: Path,
    stage_root: Path,
    *,
    release_directory: str,
    result_files: Iterable[str],
    document_files: Iterable[str],
    runtime_script_files: Iterable[str],
    test_files: Iterable[str] = (),
    result_directories: Iterable[str] = (),
    benchmark_files: Iterable[str] = (),
    benchmark_directories: Iterable[str] = (),
    closure_manifest_directory: str | None = None,
    core_benchmark_files: Iterable[str] | None = None,
    include_capture_artifacts: bool = True,
    resume: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    """Create a minimal, hash-indexed release tree for one frozen version.

    ``include_capture_artifacts=False`` produces an analysis-only package. It
    retains the hash-linked ledgers, feature-bearing manifest, results, code,
    and tests, but not source captures, vendor assets, generated HTML, raw
    provider responses, or screenshots. This is the safe default profile for
    an anonymous code release when those raw materials have separate terms.
    """
    code_root = code_root.resolve()
    stage_root = stage_root.resolve()
    release_path = _relative_path(release_directory, "release_directory")
    if len(release_path.parts) != 1:
        raise ValueError("release_directory must name one benchmark directory")
    normalized_result_files = tuple(_relative_path(path, "result file") for path in result_files)
    normalized_document_files = tuple(
        _relative_path(path, "document file") for path in document_files
    )
    normalized_runtime_scripts = runtime_script_dependency_closure(runtime_script_files)
    normalized_test_files = tuple(_relative_path(path, "test file") for path in test_files)
    normalized_result_directories = tuple(
        _relative_path(path, "result directory") for path in result_directories
    )
    normalized_benchmark_files = tuple(
        _relative_path(path, "benchmark file") for path in benchmark_files
    )
    normalized_benchmark_directories = tuple(
        _relative_path(path, "benchmark directory") for path in benchmark_directories
    )
    normalized_closure_manifest_directory = (
        _relative_path(closure_manifest_directory, "closure manifest directory")
        if closure_manifest_directory is not None
        else None
    )
    normalized_core_files = tuple(
        _relative_path(path, "core benchmark file")
        for path in (CORE_BENCHMARK_FILES if core_benchmark_files is None else core_benchmark_files)
    )
    if not normalized_core_files:
        raise ValueError("at least one core benchmark file is required")
    if stage_root.exists() and not stage_root.is_dir():
        raise ValueError(f"stage root is not a directory: {stage_root}")
    if stage_root.exists() and any(stage_root.iterdir()) and not resume:
        raise ValueError(f"stage root must be absent or empty: {stage_root}")
    if refresh and not resume:
        raise ValueError("refresh requires resume=True")

    existing_release_manifest = stage_root / "RELEASE_MANIFEST.json"
    if resume and existing_release_manifest.is_file() and not refresh:
        _remove_transient_staging_files(stage_root)
        return dict(_load_json(existing_release_manifest))
    if refresh and existing_release_manifest.is_file():
        existing_release_manifest.unlink()

    source_data_root = code_root / "benchmark" / release_path
    capture_ledger = _load_json(source_data_root / "capture_ledger.json")
    ai_records = _load_json(source_data_root / "ai_records.json")
    benchmark_manifest = _load_json(source_data_root / "webhumanbench_manifest.json")
    source_manifest = _load_json(source_data_root / "source_manifest.json")
    artifact_paths = referenced_capture_artifact_paths(capture_ledger, ai_records)

    target_code_root = stage_root / "code"
    for relative_file in ROOT_FILES:
        _copy_file(
            code_root / relative_file, target_code_root / relative_file, replace_existing=resume
        )
    for relative_file in OPTIONAL_ROOT_FILES:
        source_file = code_root / relative_file
        if source_file.is_file():
            _copy_file(source_file, target_code_root / relative_file, replace_existing=resume)
    for relative_file in normalized_document_files:
        _copy_file(
            code_root / relative_file, target_code_root / relative_file, replace_existing=resume
        )
    for relative_file in normalized_runtime_scripts:
        _copy_file(
            code_root / relative_file, target_code_root / relative_file, replace_existing=resume
        )
    for relative_file in normalized_test_files:
        _copy_file(
            code_root / relative_file, target_code_root / relative_file, replace_existing=resume
        )
    _copy_tree(
        code_root / "src" / "webmark",
        target_code_root / "src" / "webmark",
        replace_existing=resume,
    )

    target_data_root = target_code_root / "benchmark" / release_path
    for relative_file in normalized_core_files:
        _copy_file(
            source_data_root / relative_file,
            target_data_root / relative_file,
            replace_existing=resume,
        )
    for relative_file in normalized_benchmark_files:
        _copy_file(
            source_data_root / relative_file,
            target_data_root / relative_file,
            replace_existing=resume,
        )
    if include_capture_artifacts:
        _copy_paths(
            source_data_root / "captures",
            target_data_root / "captures",
            artifact_paths,
            replace_existing=resume,
        )
    if normalized_closure_manifest_directory is not None:
        source_receipts = _mapping(
            capture_ledger.get("source_receipts"), "capture ledger source_receipts"
        )
        _copy_paths(
            source_data_root / normalized_closure_manifest_directory,
            target_data_root / normalized_closure_manifest_directory,
            referenced_closure_manifest_paths(source_receipts),
            replace_existing=resume,
        )
    for relative_directory in normalized_benchmark_directories:
        _copy_tree(
            source_data_root / relative_directory,
            target_data_root / relative_directory,
            replace_existing=resume,
        )
    for relative_file in normalized_result_files:
        _copy_file(
            code_root / "results" / relative_file,
            target_code_root / "results" / relative_file,
            replace_existing=resume,
        )
    for relative_directory in normalized_result_directories:
        _copy_tree(
            code_root / "results" / relative_directory,
            target_code_root / "results" / relative_directory,
            replace_existing=resume,
        )

    metadata = _mapping(benchmark_manifest.get("metadata"), "benchmark metadata")
    source_metadata = _mapping(source_manifest.get("metadata"), "source metadata")
    if metadata.get("release_revision") != source_metadata.get("release_revision"):
        raise ValueError("source and benchmark release revisions must match")
    _remove_transient_staging_files(stage_root)
    entries = _payload_entries(stage_root)
    release_manifest = {
        "schema": RELEASE_BUNDLE_SCHEMA,
        "benchmark": metadata.get("benchmark_name"),
        "version": metadata.get("version"),
        "release_directory": release_path.as_posix(),
        "release_revision": metadata.get("release_revision"),
        "source_manifest_canonical_sha256": canonical_json_sha256(source_manifest),
        "capture_ledger_canonical_sha256": canonical_json_sha256(capture_ledger),
        "benchmark_manifest_canonical_sha256": canonical_json_sha256(benchmark_manifest),
        "referenced_capture_artifact_count": len(artifact_paths),
        "included_capture_artifact_count": len(artifact_paths) if include_capture_artifacts else 0,
        "distribution_profile": "full-provenance" if include_capture_artifacts else "analysis-only",
        "payload_file_count": len(entries),
        "payload_bytes": sum(int(entry["bytes"]) for entry in entries),
        "payload_tree_sha256": _payload_tree_sha256(entries),
        "files": entries,
        "note": (
            "This bundle excludes unreferenced candidates and transient local files. "
            "Raw source captures, vendor assets, generated HTML, provider responses, and screenshots "
            "are omitted from the analysis-only profile and remain governed by upstream/provider terms."
            if not include_capture_artifacts
            else "This bundle excludes unreferenced candidates and transient local files. Artifact rights "
            "remain governed by the release notices and upstream/provider terms."
        ),
    }
    manifest_path = stage_root / "RELEASE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _remove_transient_staging_files(stage_root)
    return release_manifest


def stage_webhumanbench_v02_release(
    code_root: Path, stage_root: Path, *, resume: bool = False, refresh: bool = False
) -> dict[str, Any]:
    """Create a clean v0.2 release tree without unrelated candidate/cache files."""
    return stage_webhumanbench_release(
        code_root,
        stage_root,
        release_directory=V02_DIRECTORY,
        result_files=RESULT_FILES,
        document_files=DOCUMENT_FILES,
        runtime_script_files=RUNTIME_SCRIPT_FILES,
        result_directories=RESULT_DIRECTORIES,
        resume=resume,
        refresh=refresh,
    )
