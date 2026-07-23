#!/usr/bin/env python3
"""Materialize bounded, fixed-commit static entrypoint closures from GitHub.

Full repository tarballs can be unnecessarily large for a static capture. This
candidate-only utility records a GitHub commit/tree binding, then writes the
declared HTML entrypoint and its reachable repository-local HTML, CSS, JS, and
asset dependencies into a loopback-only closure. Every retained file is bound
to both its Git blob SHA-1 and its SHA-256 content digest. External visual
resources remain for the separate vendored-snapshot step.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.open_reference import OpenReferenceSource, validate_open_reference_manifest
from webmark.pinned_build import SOURCE_RECEIPT_SCHEMA, assert_no_symlinks, source_tree_sha256
from webmark.release import canonical_json_sha256, sha256_file

MATERIALIZATION_METHOD = "github_entrypoint_closure_v1"
HTML_URL_RE = re.compile(r"(?:src|href)\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
CSS_URL_RE = re.compile(r"url\(\s*['\"]?([^'\"()]+)['\"]?\s*\)", re.IGNORECASE)
CSS_IMPORT_RE = re.compile(r"@import\s+(?:url\()?\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
JS_IMPORT_RE = re.compile(
    r"(?:import|export)\s+(?:[^'\"]*?\s+from\s+)?['\"]([^'\"]+)['\"]", re.IGNORECASE
)
TEXT_SUFFIXES = frozenset({".css", ".htm", ".html", ".js", ".mjs", ".cjs"})


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _github_repository(source: OpenReferenceSource) -> str:
    prefix = "https://github.com/"
    if not source.repository_url.startswith(prefix):
        raise ValueError(f"{source.id!r} is not a GitHub source")
    repository = source.repository_url.removeprefix(prefix).strip("/")
    if repository.count("/") != 1:
        raise ValueError(f"{source.id!r} repository URL is malformed")
    return repository


def _gh_api_json(endpoint: str, timeout_s: int) -> Any:
    try:
        result = subprocess.run(
            ["gh", "api", endpoint], check=False, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"gh api timed out after {timeout_s}s: {endpoint}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"gh api failed for {endpoint}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh api returned invalid JSON for {endpoint}") from exc


def _blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "path": str(row.get("path", "")),
            "mode": str(row.get("mode", "")),
            "type": str(row.get("type", "")),
            "sha": str(row.get("sha", "")),
        }
        for row in rows
    ]
    normalized.sort(key=lambda row: (row["path"], row["mode"], row["type"], row["sha"]))
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _tree_for_commit(repository: str, commit_sha: str, timeout_s: int) -> tuple[str, dict[str, str], str]:
    commit = _gh_api_json(f"repos/{repository}/git/commits/{commit_sha}", timeout_s)
    tree = commit.get("tree") if isinstance(commit, Mapping) else None
    tree_sha = tree.get("sha") if isinstance(tree, Mapping) else None
    if not isinstance(tree_sha, str) or len(tree_sha) != 40:
        raise RuntimeError("pinned commit does not resolve to a Git tree SHA")
    payload = _gh_api_json(f"repos/{repository}/git/trees/{tree_sha}?recursive=1", timeout_s)
    rows = payload.get("tree") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or payload.get("truncated"):
        raise RuntimeError("GitHub tree response is malformed or truncated")
    blobs = {
        str(row["path"]): str(row["sha"])
        for row in rows
        if isinstance(row, Mapping)
        and row.get("type") == "blob"
        and isinstance(row.get("path"), str)
        and isinstance(row.get("sha"), str)
    }
    normalized_rows = [row for row in rows if isinstance(row, Mapping)]
    return tree_sha, blobs, _tree_digest(normalized_rows)


def _blob(repository: str, blob_sha: str, timeout_s: int) -> bytes:
    payload = _gh_api_json(f"repos/{repository}/git/blobs/{blob_sha}", timeout_s)
    content = payload.get("content") if isinstance(payload, Mapping) else None
    if not isinstance(content, str) or payload.get("encoding") != "base64":
        raise RuntimeError("GitHub blob response is malformed")
    try:
        return base64.b64decode(content)
    except ValueError as exc:
        raise RuntimeError("GitHub blob response is not valid base64") from exc


def _safe_local_path(current: str, reference: str, available: Mapping[str, str]) -> str | None:
    parsed = urlsplit(reference.strip())
    if not parsed.path or parsed.scheme or parsed.netloc or reference.startswith(("#", "data:", "mailto:", "tel:")):
        return None
    raw_path = unquote(parsed.path)
    if raw_path.startswith("/"):
        candidate = raw_path.lstrip("/")
    else:
        candidate = posixpath.join(posixpath.dirname(current), raw_path)
    normalized = posixpath.normpath(candidate).lstrip("/")
    if normalized in {"", "."} or normalized.startswith("../") or "\\" in normalized:
        return None
    return normalized if normalized in available else None


def _references(path: str, data: bytes) -> list[str]:
    if PurePosixPath(path).suffix.lower() not in TEXT_SUFFIXES:
        return []
    text = data.decode("utf-8", errors="ignore")
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".htm", ".html"}:
        return HTML_URL_RE.findall(text)
    if suffix == ".css":
        return CSS_URL_RE.findall(text) + CSS_IMPORT_RE.findall(text)
    return JS_IMPORT_RE.findall(text)


def _closure_manifest(
    *, repository: str, source: OpenReferenceSource, tree_sha: str, tree_digest: str, files: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": "webmark_pinned_entrypoint_closure_v1",
        "repository": repository,
        "source_id": source.id,
        "commit_sha": source.commit_sha,
        "github_tree_sha": tree_sha,
        "github_tree_sha256": tree_digest,
        "entrypoint": source.entrypoint,
        "files": files,
        "note": (
            "Closure contains the pinned HTML entrypoint and repository-local dependencies reached by the "
            "deterministic HTML/CSS/JS URL walker. It is a capture snapshot, not a complete repository checkout."
        ),
    }


def _materialize_source(
    source: OpenReferenceSource,
    *,
    checkout_root: Path,
    closure_manifest_root: Path,
    max_files: int,
    max_bytes: int,
    timeout_s: int,
) -> dict[str, Any]:
    repository = _github_repository(source)
    destination = checkout_root / source.id
    if destination.exists():
        raise RuntimeError(f"checkout destination already exists: {destination}")
    tree_sha, blobs, tree_digest = _tree_for_commit(repository, source.commit_sha, timeout_s)
    if source.entrypoint not in blobs:
        raise RuntimeError(f"entrypoint is absent from the pinned Git tree: {source.entrypoint}")

    queue = [source.entrypoint]
    queued = {source.entrypoint}
    retained: list[dict[str, Any]] = []
    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix=f"webhumanbench-{source.id}-", dir=checkout_root) as temporary:
        temporary_root = Path(temporary) / "source"
        temporary_root.mkdir(parents=True)
        while queue:
            relative = queue.pop(0)
            blob_sha = blobs[relative]
            data = _blob(repository, blob_sha, timeout_s)
            if _blob_sha1(data) != blob_sha:
                raise RuntimeError(f"GitHub blob content does not match its declared SHA for {relative}")
            total_bytes += len(data)
            if total_bytes > max_bytes:
                raise RuntimeError(f"entrypoint closure exceeds max_bytes ({total_bytes} > {max_bytes})")
            if len(retained) >= max_files:
                raise RuntimeError(f"entrypoint closure exceeds max_files ({max_files})")
            target = temporary_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            retained.append(
                {
                    "path": relative,
                    "git_blob_sha1": blob_sha,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
            for reference in _references(relative, data):
                dependency = _safe_local_path(relative, reference, blobs)
                if dependency is not None and dependency not in queued:
                    queued.add(dependency)
                    queue.append(dependency)
        assert_no_symlinks(temporary_root)
        shutil.move(str(temporary_root), str(destination))

    retained.sort(key=lambda row: str(row["path"]))
    closure = _closure_manifest(
        repository=repository,
        source=source,
        tree_sha=tree_sha,
        tree_digest=tree_digest,
        files=retained,
    )
    closure_path = closure_manifest_root / f"{source.id}.json"
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entrypoint = destination / source.entrypoint
    return {
        "source_id": source.id,
        "capture_method": source.capture_method,
        "repository_url": source.repository_url,
        "commit_sha": source.commit_sha,
        "checkout_path": source.id,
        "checkout_tree_sha256": tree_digest,
        "checkout_file_tree_sha256": source_tree_sha256(destination),
        "entrypoint": source.entrypoint,
        "entrypoint_sha256": sha256_file(entrypoint),
        "entrypoint_git_blob_sha1": blobs[source.entrypoint],
        "checkout_bytes": total_bytes,
        "appledouble_files_removed": 0,
        "materialization_method": MATERIALIZATION_METHOD,
        "github_tree_sha": tree_sha,
        "closure_manifest_path": closure_path.name,
        "closure_manifest_sha256": sha256_file(closure_path),
        "closure_file_count": len(retained),
    }


def materialize_entrypoint_closures(
    manifest: Mapping[str, Any],
    *,
    checkout_root: Path,
    closure_manifest_root: Path,
    max_files: int,
    max_bytes: int,
    timeout_s: int,
    continue_on_error: bool,
) -> dict[str, Any]:
    """Build fixed entrypoint closures without downloading unrelated repository history."""
    if max_files <= 0 or max_bytes <= 0 or timeout_s <= 0:
        raise ValueError("max_files, max_bytes, and timeout_s must be positive")
    sources = validate_open_reference_manifest(manifest)
    checkout_root.mkdir(parents=True, exist_ok=True)
    closure_manifest_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source in sources:
        try:
            records.append(
                _materialize_source(
                    source,
                    checkout_root=checkout_root,
                    closure_manifest_root=closure_manifest_root,
                    max_files=max_files,
                    max_bytes=max_bytes,
                    timeout_s=timeout_s,
                )
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            if not continue_on_error:
                raise
            failures.append({"source_id": source.id, "error": str(exc)})
    return {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "created_at": datetime.now(UTC).isoformat(),
        "checkout_root": checkout_root.name,
        "records": records,
        "failures": failures,
        "status": "complete" if not failures else "partial_candidate_materialization",
        "note": (
            "Each record is a fixed GitHub commit/tree binding plus a bounded entrypoint dependency closure. "
            "It does not claim to be a complete repository checkout."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--closure-manifest-root", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-bytes", type=int, default=15 * 1024 * 1024)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = materialize_entrypoint_closures(
        _load_object(args.manifest),
        checkout_root=args.checkout_root,
        closure_manifest_root=args.closure_manifest_root,
        max_files=args.max_files,
        max_bytes=args.max_bytes,
        timeout_s=args.timeout_s,
        continue_on_error=args.continue_on_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"materialized {len(payload['records'])} entrypoint closures; failures: {len(payload['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
