"""Contracts shared by sandboxed fixed-commit web builds and release audits.

The build path intentionally has a narrower contract than arbitrary project
automation. A source may use only a lockfile-backed npm or Yarn Classic install
and one named package-manager script. The runner executes that script in a
filesystem-restricted and network-disabled sandbox, then records every artifact
needed to audit the captured output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

BUILD_RECIPE_SCHEMA = "webmark_pinned_local_build_recipe_v1"
BUILD_RECEIPT_SCHEMA = "webmark_pinned_local_build_receipts_v1"
BUILD_CAPTURE_METHOD = "pinned_local_build"
SOURCE_RECEIPT_SCHEMA = "webmark_pinned_source_receipts_v3"
SCRIPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
VERSION_RE = re.compile(r"v?[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9._-]+)?")
PACKAGE_MANAGER_LOCKFILES = {"npm": "package-lock.json", "yarn": "yarn.lock"}


@dataclass(frozen=True)
class BuildRecipe:
    """A deliberately small, reproducible Node build recipe."""

    raw: Mapping[str, Any]
    package_manager: str
    working_directory: str
    package_manifest: str
    lockfile: str
    build_script: str
    output_directory: str
    output_entrypoint: str
    static_export: str | None
    node_version: str
    package_manager_version: str
    install_timeout_s: int
    build_timeout_s: int

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.raw)

    @property
    def build_command(self) -> str:
        return f"{self.package_manager} run {self.build_script}"


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON semantically so formatting never changes a receipt link."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_string(value: Any, field: str, source_id: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"source {source_id!r} build_recipe.{field} must be a non-empty string")
    return text


def portable_relative_path(value: Any, field: str, source_id: str, *, allow_dot: bool = False) -> str:
    """Validate a portable path that cannot escape a source or artifact root."""
    text = _nonempty_string(value, field, source_id)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError(f"source {source_id!r} build_recipe.{field} must be a portable relative path")
    if text == "." and allow_dot:
        return text
    if text in {"", "."}:
        raise ValueError(f"source {source_id!r} build_recipe.{field} must not be the repository root")
    return text


def build_recipe_from_dict(value: Any, source_id: str) -> BuildRecipe:
    """Parse a fixed lockfile-backed build recipe without arbitrary shell commands."""
    if not isinstance(value, Mapping):
        raise ValueError(f"source {source_id!r} requires a build_recipe object for pinned_local_build")
    if value.get("schema") != BUILD_RECIPE_SCHEMA:
        raise ValueError(
            f"source {source_id!r} build_recipe.schema must be {BUILD_RECIPE_SCHEMA!r}"
        )
    package_manager = _nonempty_string(value.get("package_manager"), "package_manager", source_id)
    if package_manager not in PACKAGE_MANAGER_LOCKFILES:
        raise ValueError(
            f"source {source_id!r} build_recipe.package_manager must be one of "
            f"{sorted(PACKAGE_MANAGER_LOCKFILES)!r}"
        )
    working_directory = portable_relative_path(
        value.get("working_directory", "."), "working_directory", source_id, allow_dot=True
    )
    package_manifest = portable_relative_path(value.get("package_manifest"), "package_manifest", source_id)
    lockfile = portable_relative_path(value.get("lockfile"), "lockfile", source_id)
    expected_lockfile = PACKAGE_MANAGER_LOCKFILES[package_manager]
    if PurePosixPath(lockfile).name != expected_lockfile:
        raise ValueError(
            f"source {source_id!r} build_recipe.lockfile must name {expected_lockfile} for {package_manager}"
        )
    build_script = _nonempty_string(value.get("build_script"), "build_script", source_id)
    if not SCRIPT_RE.fullmatch(build_script):
        raise ValueError(f"source {source_id!r} build_recipe.build_script is not a portable script name")
    output_directory = portable_relative_path(value.get("output_directory"), "output_directory", source_id)
    output_entrypoint = portable_relative_path(value.get("output_entrypoint"), "output_entrypoint", source_id)
    static_export_value = value.get("static_export")
    static_export = None if static_export_value is None else _nonempty_string(
        static_export_value, "static_export", source_id
    )
    if static_export is not None:
        if static_export != "next":
            raise ValueError(f"source {source_id!r} build_recipe.static_export must be 'next' when set")
        if output_directory != "out":
            raise ValueError(
                f"source {source_id!r} build_recipe.static_export='next' requires output_directory 'out'"
            )
    node_version = _nonempty_string(value.get("node_version"), "node_version", source_id)
    npm_version = _nonempty_string(value.get("package_manager_version"), "package_manager_version", source_id)
    if not VERSION_RE.fullmatch(node_version):
        raise ValueError(f"source {source_id!r} build_recipe.node_version must be a version string")
    if not VERSION_RE.fullmatch(npm_version):
        raise ValueError(f"source {source_id!r} build_recipe.package_manager_version must be a version string")

    timeouts: dict[str, int] = {}
    for field in ("install_timeout_s", "build_timeout_s"):
        timeout = value.get(field)
        if not isinstance(timeout, int) or not 1 <= timeout <= 3_600:
            raise ValueError(f"source {source_id!r} build_recipe.{field} must be an integer in [1, 3600]")
        timeouts[field] = timeout
    return BuildRecipe(
        raw=dict(value),
        package_manager=package_manager,
        working_directory=working_directory,
        package_manifest=package_manifest,
        lockfile=lockfile,
        build_script=build_script,
        output_directory=output_directory,
        output_entrypoint=output_entrypoint,
        static_export=static_export,
        node_version=node_version,
        package_manager_version=npm_version,
        install_timeout_s=timeouts["install_timeout_s"],
        build_timeout_s=timeouts["build_timeout_s"],
    )


def _safe_artifact_path(value: Any, field: str) -> PurePosixPath:
    text = str(value).strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError(f"{field} must be a portable relative path")
    return path


def _sha256(value: Any, field: str) -> str:
    text = str(value).strip()
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _artifact_digest(value: Any, field: str, artifact_root: Path | None) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    path = _safe_artifact_path(value.get("path"), f"{field}.path")
    digest = _sha256(value.get("sha256"), f"{field}.sha256")
    if artifact_root is not None:
        resolved = artifact_root / path
        if not resolved.is_file():
            raise ValueError(f"{field}.path does not exist under artifact_root: {path}")
        if sha256_file(resolved) != digest:
            raise ValueError(f"{field}.sha256 does not match {path}")
    return digest


def _assert_regular_tree(root: Path) -> None:
    """Reject symlinks and special files before hashing or serving a build output."""
    if not root.is_dir():
        raise ValueError(f"build output directory does not exist: {root}")
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"build output contains a symlink: {path}")
        if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
            raise ValueError(f"build output contains a non-regular file: {path}")


def file_tree_manifest(root: Path, *, exclude_top_level: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Return a deterministic, symlink-free manifest of a regular file tree."""
    _assert_regular_tree(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in exclude_top_level:
            continue
        rows.append({
            "path": relative.as_posix(),
            "mode": stat.S_IMODE(path.stat().st_mode),
            "sha256": sha256_file(path),
        })
    if not rows:
        raise ValueError(f"file tree is empty: {root}")
    return rows


def output_tree_manifest(root: Path) -> list[dict[str, Any]]:
    """Return a deterministic, symlink-free manifest of a built static tree."""
    return file_tree_manifest(root)


def output_tree_sha256(root: Path) -> str:
    return canonical_json_sha256(output_tree_manifest(root))


def source_tree_sha256(root: Path) -> str:
    """Hash a materialized source tree while excluding mutable Git metadata."""
    return canonical_json_sha256(file_tree_manifest(root, exclude_top_level=frozenset({".git"})))


def _source_by_id(sources: Sequence[Any]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for source in sources:
        source_id = str(source.id)
        if source_id in indexed:
            raise ValueError(f"duplicate source ID in build-receipt validation: {source_id!r}")
        indexed[source_id] = source
    return indexed


def validate_build_receipts(
    payload: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    sources: Sequence[Any],
    *,
    artifact_root: Path | None,
) -> dict[str, Mapping[str, Any]]:
    """Validate build receipt links and, optionally, every retained build artifact.

    ``sources`` are already validated source rows.  Keeping this function free
    of an ``open_reference`` import avoids a circular dependency while allowing
    both the capture script and public-release audit to enforce the same
    immutable-output contract.
    """
    if payload.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise ValueError(f"build receipts schema must be {BUILD_RECEIPT_SCHEMA!r}")
    if payload.get("source_manifest_sha256") != canonical_json_sha256(source_manifest):
        raise ValueError("build receipts source_manifest_sha256 does not match the source manifest")
    source_by_id = _source_by_id(sources)
    expected_ids = {
        source_id
        for source_id, source in source_by_id.items()
        if getattr(source, "capture_method", None) == BUILD_CAPTURE_METHOD
    }
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("build receipts requires a non-empty records list")
    records: dict[str, Mapping[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("build receipt records must be objects")
        source_id = str(raw_record.get("source_id", "")).strip()
        if not source_id or source_id in records:
            raise ValueError("build receipt source_id values must be non-empty and unique")
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"build receipt references unknown source_id {source_id!r}")
        if getattr(source, "capture_method", None) != BUILD_CAPTURE_METHOD:
            raise ValueError(f"build receipt source {source_id!r} is not a pinned_local_build source")
        recipe = getattr(source, "build_recipe", None)
        if recipe is None:
            raise ValueError(f"build receipt source {source_id!r} has no validated build recipe")
        if raw_record.get("commit_sha") != source.commit_sha:
            raise ValueError(f"build receipt {source_id!r} commit_sha does not match its source")
        if raw_record.get("entrypoint") != source.entrypoint:
            raise ValueError(f"build receipt {source_id!r} entrypoint does not match its source")
        _sha256(raw_record.get("checkout_tree_sha256"), f"build receipt {source_id!r} checkout_tree_sha256")
        _sha256(raw_record.get("entrypoint_sha256"), f"build receipt {source_id!r} entrypoint_sha256")
        source_materialization = raw_record.get("source_materialization")
        if not isinstance(source_materialization, Mapping):
            raise ValueError(f"build receipt {source_id!r} source_materialization must be an object")
        materialization_method = source_materialization.get("materialization_method")
        if materialization_method not in {"git_checkout", "github_api_tarball"}:
            raise ValueError(f"build receipt {source_id!r} has an unsupported source materialization method")
        if source_materialization.get("checkout_tree_sha256") != raw_record.get("checkout_tree_sha256"):
            raise ValueError(f"build receipt {source_id!r} source materialization tree does not match")
        _sha256(
            source_materialization.get("checkout_file_tree_sha256"),
            f"build receipt {source_id!r} checkout_file_tree_sha256",
        )
        if materialization_method == "github_api_tarball":
            github_tree_sha = str(source_materialization.get("github_tree_sha", ""))
            if not re.fullmatch(r"[0-9a-f]{40}", github_tree_sha):
                raise ValueError(f"build receipt {source_id!r} needs a pinned GitHub tree SHA")
            _sha256(
                source_materialization.get("source_archive_sha256"),
                f"build receipt {source_id!r} source_archive_sha256",
            )
        if raw_record.get("build_recipe_sha256") != recipe.sha256:
            raise ValueError(f"build receipt {source_id!r} build_recipe_sha256 does not match its source")
        if raw_record.get("node_version") != recipe.node_version:
            raise ValueError(f"build receipt {source_id!r} node_version does not match its recipe")
        if raw_record.get("package_manager") != recipe.package_manager:
            raise ValueError(f"build receipt {source_id!r} package_manager does not match its recipe")
        if raw_record.get("package_manager_version") != recipe.package_manager_version:
            raise ValueError(f"build receipt {source_id!r} package_manager_version does not match its recipe")
        if raw_record.get("lockfile") != recipe.lockfile:
            raise ValueError(f"build receipt {source_id!r} lockfile does not match its recipe")
        _sha256(raw_record.get("lockfile_sha256"), f"build receipt {source_id!r} lockfile_sha256")
        _sha256(raw_record.get("package_manifest_sha256"), f"build receipt {source_id!r} package_manifest_sha256")
        if raw_record.get("output_entrypoint") != recipe.output_entrypoint:
            raise ValueError(f"build receipt {source_id!r} output_entrypoint does not match its recipe")
        if raw_record.get("static_export") != recipe.static_export:
            raise ValueError(f"build receipt {source_id!r} static_export does not match its recipe")
        output_path = _safe_artifact_path(
            raw_record.get("output_directory"), f"build receipt {source_id!r} output_directory"
        )
        expected_output_path = PurePosixPath("builds") / source_id / "output"
        if output_path != expected_output_path:
            raise ValueError(f"build receipt {source_id!r} output_directory must be {expected_output_path}")
        _sha256(raw_record.get("output_tree_sha256"), f"build receipt {source_id!r} output_tree_sha256")
        _sha256(
            raw_record.get("output_entrypoint_sha256"),
            f"build receipt {source_id!r} output_entrypoint_sha256",
        )
        sandbox = raw_record.get("sandbox")
        if not isinstance(sandbox, Mapping):
            raise ValueError(f"build receipt {source_id!r} sandbox must be an object")
        if sandbox.get("backend") != "sandbox-exec":
            raise ValueError(f"build receipt {source_id!r} must use sandbox-exec")
        if sandbox.get("install_lifecycle_scripts") is not False:
            raise ValueError(f"build receipt {source_id!r} must disable install lifecycle scripts")
        if sandbox.get("build_network") != "denied":
            raise ValueError(f"build receipt {source_id!r} must deny network during the build")
        _sha256(sandbox.get("install_profile_sha256"), f"build receipt {source_id!r} install profile")
        _sha256(sandbox.get("build_profile_sha256"), f"build receipt {source_id!r} build profile")
        artifacts = raw_record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"build receipt {source_id!r} artifacts must be an object")
        for name in ("install_log", "build_log", "output_manifest"):
            _artifact_digest(artifacts.get(name), f"build receipt {source_id!r} artifacts.{name}", artifact_root)
        if artifact_root is not None:
            output_root = artifact_root / output_path
            observed_manifest = output_tree_manifest(output_root)
            observed_tree = canonical_json_sha256(observed_manifest)
            if observed_tree != raw_record.get("output_tree_sha256"):
                raise ValueError(f"build receipt {source_id!r} output_tree_sha256 does not match its output tree")
            entrypoint = output_root / recipe.output_entrypoint
            if not entrypoint.is_file() or sha256_file(entrypoint) != raw_record.get("output_entrypoint_sha256"):
                raise ValueError(f"build receipt {source_id!r} output entrypoint digest does not match")
            output_manifest_artifact = artifacts.get("output_manifest")
            if not isinstance(output_manifest_artifact, Mapping):
                raise ValueError(f"build receipt {source_id!r} output_manifest artifact must be an object")
            manifest_path = artifact_root / _safe_artifact_path(
                output_manifest_artifact.get("path"),
                f"build receipt {source_id!r} artifacts.output_manifest.path",
            )
            try:
                with manifest_path.open(encoding="utf-8") as handle:
                    saved_manifest = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"build receipt {source_id!r} output manifest must be valid JSON") from exc
            if saved_manifest != observed_manifest:
                raise ValueError(f"build receipt {source_id!r} output manifest does not match its output tree")
        records[source_id] = raw_record
    if set(records) != expected_ids:
        missing = sorted(expected_ids.difference(records))
        extra = sorted(set(records).difference(expected_ids))
        raise ValueError(f"build receipts do not match build sources; missing={missing[:3]}, extra={extra[:3]}")
    return records


def sandbox_profile(work_directory: Path, *, allow_network: bool) -> str:
    """Create a macOS Seatbelt profile for a hermetic build work directory.

    ``sandbox-exec`` needs a small set of macOS runtime allowances.  The
    profile imports Apple's system baseline, switches back to deny-by-default,
    then explicitly grants reads to runtime roots and the copied source tree.
    The caller also supplies a scrubbed environment, so project scripts cannot
    inherit host credentials through environment variables.
    """
    work = str(work_directory.resolve())
    allowed_read_roots = (
        work,
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/Library",
        "/opt/homebrew",
        "/dev",
        "/etc",
        "/private/etc",
        "/private/var/db",
        "/private/var/run",
    )
    allowed_reads = " ".join(f'(subpath "{root}")' for root in allowed_read_roots)
    parts = [
        "(version 1)",
        '(import "system.sb")',
        "(deny default)",
        "(allow process*)",
        '(allow file-read* file-test-existence (literal "/opt") (literal "/private") (literal "/private/var"))',
        f'(allow file-read-metadata file-test-existence (path-ancestors "{work}"))',
        f"(allow file-read* file-test-existence {allowed_reads})",
        f'(allow file-write* (subpath "{work}"))',
        '(allow file-write-data (literal "/dev/null") (literal "/dev/zero"))',
    ]
    if allow_network:
        parts.append("(allow network*)")
    else:
        parts.append("(deny network*)")
    return " ".join(parts)


def scrubbed_build_environment(work_directory: Path) -> dict[str, str]:
    """Return the minimal environment used for sandboxed npm subprocesses."""
    home = work_directory / "home"
    temp = work_directory / "tmp"
    cache = work_directory / "npm-cache"
    for directory in (home, temp, cache):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "CI": "true",
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "NO_UPDATE_NOTIFIER": "true",
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(temp),
        "npm_config_audit": "false",
        "npm_config_cache": str(cache),
        "npm_config_fund": "false",
        "npm_config_update_notifier": "false",
        "npm_config_userconfig": str(home / ".npmrc"),
        "YARN_CACHE_FOLDER": str(cache / "yarn"),
        "YARN_ENABLE_TELEMETRY": "0",
    }


def assert_no_symlinks(root: Path) -> None:
    """Reject source inputs that could make a build escape its copied tree."""
    if not root.is_dir():
        raise ValueError(f"source directory does not exist: {root}")
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"source tree contains a symlink: {path}")
