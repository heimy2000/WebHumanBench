#!/usr/bin/env python3
"""Fail closed on files that should not enter a public WebHumanBench release.

The source/capture/benchmark audit verifies the declared corpus. This separate
preflight scans a staged release tree for local dependency directories, cache
files, AppleDouble metadata, oversized files, symlinks, credential-like files,
and common token formats. It is deliberately conservative: review every finding
instead of publishing from a working directory by accident.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

PREFLIGHT_SCHEMA = "webmark_public_release_preflight_v1"
FORBIDDEN_DIRECTORIES = frozenset({
    ".git",
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
})
SENSITIVE_FILENAMES = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
})
SENSITIVE_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx"})
TOKEN_PATTERNS = (
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def _issue(kind: str, path: Path, detail: str) -> dict[str, str]:
    return {"kind": kind, "path": path.as_posix(), "detail": detail}


def _contains_token(path: Path) -> bool:
    """Check a bounded text prefix without including a possible secret in output."""
    try:
        with path.open("rb") as handle:
            sample = handle.read(1024 * 1024)
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return any(pattern.search(sample) for pattern in TOKEN_PATTERNS)


def _is_forbidden_directory(name: str) -> bool:
    return name in FORBIDDEN_DIRECTORIES or name.endswith(".dist-info") or name.endswith(".egg-info")


def preflight_public_release(root: Path, *, max_file_size_bytes: int = 50 * 1024 * 1024) -> dict[str, Any]:
    """Return a deterministic, fail-closed report for a staged release tree."""
    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"release root is not a directory: {root}")

    issues: list[dict[str, str]] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                issues.append(_issue("symlink", relative, "directories must be materialized in a public archive"))
                continue
            if _is_forbidden_directory(name):
                issues.append(_issue("forbidden_directory", relative, "local dependency or cache directory"))
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                issues.append(_issue("symlink", relative, "files must be materialized in a public archive"))
                continue
            if name == ".DS_Store" or name.startswith("._"):
                issues.append(_issue("os_metadata", relative, "remove operating-system metadata"))
                continue
            if name == ".coverage":
                issues.append(_issue("cache_file", relative, "remove local coverage output"))
                continue
            if name in SENSITIVE_FILENAMES or candidate.suffix.lower() in SENSITIVE_SUFFIXES:
                issues.append(_issue("sensitive_filename", relative, "remove credentials or private-key material"))
                continue
            try:
                size = candidate.stat().st_size
            except OSError as exc:
                issues.append(_issue("unreadable_file", relative, str(exc)))
                continue
            if size > max_file_size_bytes:
                issues.append(
                    _issue(
                        "oversized_file",
                        relative,
                        f"{size} bytes exceeds configured {max_file_size_bytes}-byte limit",
                    )
                )
            if _contains_token(candidate):
                issues.append(_issue("credential_pattern", relative, "common access-token format detected"))

    issues.sort(key=lambda issue: (issue["path"], issue["kind"], issue["detail"]))
    return {
        "schema": PREFLIGHT_SCHEMA,
        "root": root.name,
        "max_file_size_bytes": max_file_size_bytes,
        "status": "pass" if not issues else "fail",
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--max-file-size-mb", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_file_size_mb <= 0:
        parser.error("--max-file-size-mb must be positive")
    max_file_size_bytes = round(args.max_file_size_mb * 1024 * 1024)
    report = preflight_public_release(args.release_root, max_file_size_bytes=max_file_size_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if report["status"] == "pass":
        print("public release preflight: passed")
        return 0
    print(f"public release preflight: failed ({len(report['issues'])} findings)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
