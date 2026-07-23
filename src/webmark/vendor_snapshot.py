"""Validation helpers for vendored static-source capture overlays."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .pinned_build import canonical_json_sha256, sha256_file

SNAPSHOT_CAPTURE_METHOD = "pinned_local_static_snapshot"
VENDOR_RECEIPT_SCHEMA = "webmark_vendored_static_snapshot_receipts_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sha256(value: Any, field: str) -> str:
    text = str(value).strip()
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _relative_path(value: Any, field: str) -> PurePosixPath:
    text = str(value).strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError(f"{field} must be a portable relative path")
    return path


def _artifact(value: Any, field: str, artifact_root: Path | None) -> dict[str, str]:
    item = _mapping(value, field)
    path = _relative_path(item.get("path"), f"{field}.path")
    digest = _sha256(item.get("sha256"), f"{field}.sha256")
    if artifact_root is not None:
        resolved = artifact_root / path
        if not resolved.is_file():
            raise ValueError(f"{field}.path does not exist under artifact_root: {path}")
        if sha256_file(resolved) != digest:
            raise ValueError(f"{field}.sha256 does not match {path}")
    return {"path": path.as_posix(), "sha256": digest}


def _https_url(value: Any, field: str) -> str:
    text = str(value).strip()
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{field} must be an HTTPS URL without credentials")
    return text


def validate_vendor_receipts(
    payload: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    sources: Sequence[Any],
    *,
    artifact_root: Path | None,
) -> dict[str, Mapping[str, Any]]:
    """Validate frozen third-party visual assets used by snapshot captures."""
    if payload.get("schema") != VENDOR_RECEIPT_SCHEMA:
        raise ValueError(f"vendor receipts schema must be {VENDOR_RECEIPT_SCHEMA!r}")
    if payload.get("source_manifest_sha256") != canonical_json_sha256(source_manifest):
        raise ValueError("vendor receipts source_manifest_sha256 does not match source manifest")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("vendor receipts requires a records list")
    expected = {source.id: source for source in sources if source.capture_method == SNAPSHOT_CAPTURE_METHOD}
    records: dict[str, Mapping[str, Any]] = {}
    for raw_row in rows:
        row = _mapping(raw_row, "vendor receipt record")
        source_id = str(row.get("source_id", "")).strip()
        source = expected.get(source_id)
        if source is None or source_id in records:
            raise ValueError("vendor receipt source_id values must identify one snapshot source each")
        if row.get("commit_sha") != source.commit_sha or row.get("entrypoint") != source.entrypoint:
            raise ValueError(f"vendor receipt {source_id!r} does not match its pinned source")
        _sha256(row.get("original_entrypoint_sha256"), f"vendor receipt {source_id!r} original_entrypoint_sha256")
        overlay = _artifact(row.get("overlay_entrypoint"), f"vendor receipt {source_id!r} overlay_entrypoint", artifact_root)
        if row.get("snapshot_entrypoint_sha256") != overlay["sha256"]:
            raise ValueError(f"vendor receipt {source_id!r} snapshot_entrypoint_sha256 does not match overlay")
        overlays = row.get("overlay_files")
        if not isinstance(overlays, list) or not overlays:
            raise ValueError(f"vendor receipt {source_id!r} requires non-empty overlay_files")
        overlay_files = [_artifact(item, f"vendor receipt {source_id!r} overlay_files", artifact_root) for item in overlays]
        if overlay not in overlay_files:
            raise ValueError(f"vendor receipt {source_id!r} overlay_files must include overlay_entrypoint")
        assets = row.get("vendor_assets")
        if not isinstance(assets, list):
            raise ValueError(f"vendor receipt {source_id!r} vendor_assets must be a list")
        asset_urls: set[str] = set()
        for index, raw_asset in enumerate(assets):
            asset = _mapping(raw_asset, f"vendor receipt {source_id!r} vendor_assets[{index}]")
            url = _https_url(asset.get("original_url"), f"vendor receipt {source_id!r} vendor asset URL")
            if url in asset_urls:
                raise ValueError(f"vendor receipt {source_id!r} has duplicate asset URL")
            asset_urls.add(url)
            _artifact(asset.get("artifact"), f"vendor receipt {source_id!r} vendor asset", artifact_root)
            if not isinstance(asset.get("content_type"), str) or not asset["content_type"]:
                raise ValueError(f"vendor receipt {source_id!r} vendor asset requires content_type")
        removed = row.get("removed_external_scripts")
        if not isinstance(removed, list):
            raise ValueError(f"vendor receipt {source_id!r} removed_external_scripts must be a list")
        for index, raw_removed in enumerate(removed):
            item = _mapping(raw_removed, f"vendor receipt {source_id!r} removed_external_scripts[{index}]")
            _https_url(item.get("url"), f"vendor receipt {source_id!r} removed external script URL")
            if not isinstance(item.get("reason"), str) or not item["reason"]:
                raise ValueError(f"vendor receipt {source_id!r} removed external script needs a reason")
        records[source_id] = row
    if set(records) != set(expected):
        missing = sorted(set(expected).difference(records))
        extra = sorted(set(records).difference(expected))
        raise ValueError(f"vendor receipts do not match snapshot sources; missing={missing[:3]}, extra={extra[:3]}")
    return records
