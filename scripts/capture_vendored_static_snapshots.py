#!/usr/bin/env python3
"""Capture fixed static sources after freezing their visual third-party assets.

The source checkout remains immutable.  A temporary overlay rewrites visual
HTTPS assets to content-addressed local copies and excludes a narrow set of
nonvisual tracking and legacy browser-compatibility scripts. The overlay and
every fetched asset are retained as hash-addressed artifacts, then the browser
is forced offline except for the loopback snapshot server.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import importlib.metadata
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capture_open_mobile_reference import (  # noqa: E402
    CAPTURE_SCHEMA,
    DEFAULT_CHROME,
    FEATURE_EXTRACTOR_VERSION,
    _artifact_stem,
    _capture_protocol,
    _computed_feature_script,
    _load_static_checkout_receipts,
    _local_only_route,
    _viewport,
)

from webmark.open_reference import (  # noqa: E402
    OpenReferenceSource,
    validate_open_reference_manifest,
)
from webmark.release import canonical_json_sha256, sha256_file  # noqa: E402
from webmark.vendor_snapshot import SNAPSHOT_CAPTURE_METHOD, VENDOR_RECEIPT_SCHEMA  # noqa: E402

EXCLUDED_EXTERNAL_SCRIPT_REASONS = {
    "www.googletagmanager.com": "nonvisual_analytics_or_tracking_script",
    "www.google-analytics.com": "nonvisual_analytics_or_tracking_script",
    "analytics.google.com": "nonvisual_analytics_or_tracking_script",
    "s7.addthis.com": "nonvisual_analytics_or_tracking_script",
    # html5shiv exists only to teach obsolete Internet Explorer versions about
    # HTML5 elements. It has no visual role in the fixed Chromium protocol.
    "html5shiv.googlecode.com": "legacy_browser_compatibility_script",
}
ASSET_TIMEOUT_S = 30
MAX_ASSET_BYTES = 10 * 1024 * 1024
ASSET_FETCH_ATTEMPTS = 3
TAG_RE = re.compile(
    r"<(?P<tag>link|script|img|source|video|audio|iframe)\b(?P<attrs>[^>]*)>", re.IGNORECASE
)
URL_ATTR_RE = re.compile(
    r"(?P<name>src|href)\s*=\s*(?P<quote>[\"'])(?P<url>(?:https?:)?//[^\"']+)(?P=quote)",
    re.IGNORECASE,
)
EXTERNAL_SCRIPT_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*(?P<quote>[\"'])(?P<url>(?:https?:)?//[^\"']+)(?P=quote)[^>]*>\s*</script>",
    re.IGNORECASE,
)
CSS_URL_RE = re.compile(r"url\(\s*(?P<quote>[\"']?)(?P<url>[^\"')]+)(?P=quote)\s*\)", re.IGNORECASE)


def _artifact(path: Path, artifact_root: Path) -> dict[str, str]:
    return {"path": path.relative_to(artifact_root).as_posix(), "sha256": sha256_file(path)}


def _asset_suffix(url: str, content_type: str) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix and 1 < len(suffix) <= 8 and suffix[1:].isalnum():
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return guessed or ".bin"


def _relative_url(source: Path, target: Path) -> str:
    return Path(os.path.relpath(target, source.parent)).as_posix()


def _excluded_external_script_reason(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return EXCLUDED_EXTERNAL_SCRIPT_REASONS.get(host)


def _canonical_external_script_url(url: str) -> str:
    """Record removed external scripts with the HTTPS form required by releases."""
    if url.startswith("//"):
        return f"https:{url}"
    parsed = urlparse(url)
    if parsed.scheme == "http":
        return parsed._replace(scheme="https").geturl()
    return url


class _VendoredSnapshot:
    """A disposable source overlay with frozen visual external assets."""

    def __init__(
        self,
        source: OpenReferenceSource,
        checkout: Path,
        artifact_root: Path,
        *,
        timeout_s: int,
        max_asset_bytes: int,
    ) -> None:
        self.source = source
        self.checkout = checkout
        self.artifact_root = artifact_root
        self.timeout_s = timeout_s
        self.max_asset_bytes = max_asset_bytes
        self._temporary = tempfile.TemporaryDirectory(prefix=f"webhumanbench-snapshot-{source.id}-")
        self.root = Path(self._temporary.name) / "source"
        self.vendor_root = self.root / "_webhumanbench_vendor"
        self.entrypoint = self.root / source.entrypoint
        self.original_entrypoint_sha256 = sha256_file(checkout / source.entrypoint)
        self.assets: dict[str, dict[str, Any]] = {}
        self.removed_scripts: list[dict[str, str]] = []
        self.modified_files: set[Path] = set()

    def __enter__(self) -> _VendoredSnapshot:
        shutil.copytree(self.checkout, self.root)
        if not self.entrypoint.is_file():
            raise RuntimeError(f"snapshot source entrypoint is absent: {self.source.entrypoint}")
        self.vendor_root.mkdir()
        self._rewrite_html(self.entrypoint)
        # Retain the exact entrypoint even when no external resource needed rewriting.
        self.modified_files.add(self.entrypoint)
        for css_path in sorted(self.root.rglob("*.css")):
            if self.vendor_root not in css_path.parents:
                self._rewrite_css(css_path, base_url=None)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self._temporary.cleanup()

    def _fetch_asset(self, url: str) -> dict[str, Any]:
        existing = self.assets.get(url)
        if existing is not None:
            return existing
        request = Request(
            url, headers={"User-Agent": "Mozilla/5.0 (WebHumanBench snapshot freezer)"}
        )
        last_error: OSError | URLError | None = None
        content_type: str
        data: bytes
        for attempt in range(1, ASSET_FETCH_ATTEMPTS + 1):
            try:
                with urlopen(  # nosec B310 - source URL is recorded and HTTPS-only
                    request, timeout=self.timeout_s
                ) as response:
                    content_type = response.headers.get_content_type() or "application/octet-stream"
                    data = response.read(self.max_asset_bytes + 1)
                break
            except HTTPError:
                # A stable HTTP error is evidence that the URL cannot be frozen;
                # retrying it would only make release results less deterministic.
                raise
            except (OSError, URLError) as exc:
                last_error = exc
                if attempt == ASSET_FETCH_ATTEMPTS:
                    raise
                time.sleep(0.25 * attempt)
        else:  # pragma: no cover - loop either breaks or raises
            assert last_error is not None
            raise last_error
        if len(data) > self.max_asset_bytes:
            raise RuntimeError(f"external asset exceeds {self.max_asset_bytes} bytes: {url}")
        filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + _asset_suffix(
            url, content_type
        )
        path = self.vendor_root / filename
        path.write_bytes(data)
        asset = {"original_url": url, "path": path, "content_type": content_type}
        self.assets[url] = asset
        if content_type == "text/css" or path.suffix.lower() == ".css":
            self._rewrite_css(path, base_url=url)
        return asset

    def _vendor_url(self, url: str, source_path: Path, *, base_url: str | None) -> str:
        if url.startswith(("data:", "#")):
            return url
        # A local HTTP snapshot would otherwise resolve ``//host/path`` over
        # HTTP. Freeze protocol-relative visual assets through HTTPS instead.
        absolute = (
            f"https:{url}" if url.startswith("//") else urljoin(base_url, url) if base_url else url
        )
        parsed_absolute = urlparse(absolute)
        if parsed_absolute.scheme not in {"http", "https"}:
            return url
        if parsed_absolute.scheme != "https":
            raise RuntimeError(f"refusing non-HTTPS external asset: {absolute}")
        asset = self._fetch_asset(absolute)
        return _relative_url(source_path, Path(asset["path"]))

    def _rewrite_css(self, path: Path, *, base_url: str | None) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return

        def replace(match: re.Match[str]) -> str:
            url = match.group("url").strip()
            replacement = self._vendor_url(url, path, base_url=base_url)
            quote = match.group("quote") or ""
            return f"url({quote}{replacement}{quote})"

        rewritten = CSS_URL_RE.sub(replace, text)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            if self.vendor_root not in path.parents:
                self.modified_files.add(path)

    def _rewrite_html(self, path: Path) -> None:
        original_html = path.read_text(encoding="utf-8")
        html = original_html

        def remove_nonvisual(match: re.Match[str]) -> str:
            url = match.group("url")
            reason = _excluded_external_script_reason(url)
            if reason is not None:
                self.removed_scripts.append(
                    {"url": _canonical_external_script_url(url), "reason": reason}
                )
                return ""
            return match.group(0)

        html = EXTERNAL_SCRIPT_RE.sub(remove_nonvisual, html)

        def replace_tag(match: re.Match[str]) -> str:
            tag = match.group("tag").lower()
            attrs = match.group("attrs")
            if tag == "link" and re.search(
                r"\brel\s*=\s*([\"'])?canonical\1", attrs, re.IGNORECASE
            ):
                return match.group(0)

            def replace_attr(attr_match: re.Match[str]) -> str:
                url = attr_match.group("url")
                replacement = self._vendor_url(url, path, base_url=None)
                return f"{attr_match.group('name')}={attr_match.group('quote')}{replacement}{attr_match.group('quote')}"

            return f"<{tag}{URL_ATTR_RE.sub(replace_attr, attrs)}>"

        rewritten = TAG_RE.sub(replace_tag, html)
        if rewritten != original_html:
            path.write_text(rewritten, encoding="utf-8")
            self.modified_files.add(path)

    def receipt(self) -> dict[str, Any]:
        artifact_root = self.artifact_root
        destination = artifact_root / "snapshot_overlays" / self.source.id
        overlay_root = destination / "overlay"
        vendor_destination = destination / "vendor"
        overlay_files: list[dict[str, str]] = []
        for path in sorted(self.modified_files):
            relative = path.relative_to(self.root)
            target = overlay_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            overlay_files.append(_artifact(target, artifact_root))
        vendor_assets: list[dict[str, Any]] = []
        for url, asset in sorted(self.assets.items()):
            source_path = Path(asset["path"])
            target = vendor_destination / source_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            vendor_assets.append(
                {
                    "original_url": url,
                    "content_type": asset["content_type"],
                    "artifact": _artifact(target, artifact_root),
                }
            )
        overlay_entrypoint = overlay_root / self.source.entrypoint
        if not overlay_entrypoint.is_file():
            raise RuntimeError("snapshot receipt is missing its overlay entrypoint")
        return {
            "source_id": self.source.id,
            "commit_sha": self.source.commit_sha,
            "entrypoint": self.source.entrypoint,
            "original_entrypoint_sha256": self.original_entrypoint_sha256,
            "snapshot_entrypoint_sha256": sha256_file(overlay_entrypoint),
            "overlay_entrypoint": _artifact(overlay_entrypoint, artifact_root),
            "overlay_files": overlay_files,
            "vendor_assets": vendor_assets,
            "removed_external_scripts": self.removed_scripts,
            "recipe": {
                "mode": "freeze_https_visual_assets_v1",
                "nonvisual_script_policy": "remove_known_nonvisual_or_legacy_compatibility_hosts_v2",
                "asset_fetch_attempts": ASSET_FETCH_ATTEMPTS,
                "max_asset_bytes": self.max_asset_bytes,
            },
        }


class _StaticServer:
    def __init__(self, directory: Path) -> None:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> _StaticServer:
        self.thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _capture_snapshot(
    page: Any,
    source: OpenReferenceSource,
    viewport: str,
    target_url: str,
    artifact_root: Path,
    capture_origin: Mapping[str, Any],
    timeout_ms: int,
) -> dict[str, Any]:
    page.set_viewport_size(_viewport(viewport))
    response = page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(500)
    features = page.evaluate(_computed_feature_script())
    if not all(
        features.get(name) for name in ("typography", "spacing", "grid", "color", "saturation")
    ):
        raise RuntimeError(
            f"{source.id} at {viewport} produced an incomplete computed feature vector"
        )
    stem = _artifact_stem(source.id, viewport)
    html_path = artifact_root / "html" / f"{stem}.html"
    screenshot_path = artifact_root / "screenshots" / f"{stem}.png"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page.content(), encoding="utf-8")
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "id": f"{source.id}@{viewport}",
        "source_id": source.id,
        "source": "human",
        "group_id": source.group_id,
        "page_type": source.page_type,
        "viewport": viewport,
        "capture_url": source.capture_url,
        "final_url": page.url,
        "http_status": response.status if response is not None else None,
        "capture_origin": dict(capture_origin),
        "captured_at": datetime.now(UTC).isoformat(),
        "features": features,
        "feature_sha256": canonical_json_sha256(features),
        "capture_html_sha256": sha256_file(html_path),
        "artifacts": {
            "html": _artifact(html_path, artifact_root),
            "screenshot": _artifact(screenshot_path, artifact_root),
        },
    }


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def capture_vendored_static_snapshots(
    manifest: Mapping[str, Any],
    *,
    checkout_root: Path,
    checkout_receipts_path: Path,
    artifact_root: Path,
    chrome: Path,
    timeout_ms: int,
    vendor_timeout_s: int,
    max_asset_bytes: int,
    continue_on_error: bool,
    allow_partial_receipts: bool = False,
) -> dict[str, Any]:
    """Freeze external visual assets, then capture every snapshot source locally."""
    sources = validate_open_reference_manifest(manifest)
    if any(source.capture_method != SNAPSHOT_CAPTURE_METHOD for source in sources):
        raise ValueError(
            "vendored snapshot capture accepts only pinned_local_static_snapshot sources"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("source manifest requires metadata")
    protocol = _capture_protocol(metadata)
    if timeout_ms <= 0 or vendor_timeout_s <= 0 or max_asset_bytes <= 0:
        raise ValueError("capture and vendoring limits must be positive")
    if allow_partial_receipts and not continue_on_error:
        raise ValueError("allow_partial_receipts requires continue_on_error")
    receipts = _load_static_checkout_receipts(
        checkout_receipts_path,
        manifest,
        sources,
        checkout_root,
        allow_partial=allow_partial_receipts,
    )
    source_receipts = _load_object(checkout_receipts_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError(
            'Install browser dependencies with: pip install -e ".[browser]"'
        ) from exc

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    vendor_records: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=str(chrome) if chrome.exists() else None
        )
        try:
            browser_version = browser.version
            if protocol["browser_version"] != browser_version:
                raise RuntimeError(
                    "capture_protocol.browser_version does not match launched browser: "
                    f"expected {protocol['browser_version']!r}, got {browser_version!r}"
                )
            for source in sources:
                if source.id not in receipts:
                    failures.extend(
                        {
                            "source_id": source.id,
                            "viewport": viewport,
                            "error": "source was not materialized into the candidate receipt",
                        }
                        for viewport in source.viewports
                    )
                    continue
                receipt = receipts[source.id]
                checkout = checkout_root / str(receipt["checkout_path"])
                try:
                    with _VendoredSnapshot(
                        source,
                        checkout,
                        artifact_root,
                        timeout_s=vendor_timeout_s,
                        max_asset_bytes=max_asset_bytes,
                    ) as snapshot:
                        vendor_receipt = snapshot.receipt()
                        vendor_records.append(vendor_receipt)
                        origin = {
                            "mode": SNAPSHOT_CAPTURE_METHOD,
                            "commit_sha": source.commit_sha,
                            "checkout_tree_sha256": receipt["checkout_tree_sha256"],
                            "entrypoint_sha256": receipt["entrypoint_sha256"],
                            "source_receipt_sha256": canonical_json_sha256(receipt),
                            "source_materialization_method": receipt["materialization_method"],
                            "closure_manifest_sha256": receipt["closure_manifest_sha256"],
                            "vendor_receipt_sha256": canonical_json_sha256(vendor_receipt),
                            "snapshot_entrypoint_sha256": vendor_receipt[
                                "snapshot_entrypoint_sha256"
                            ],
                        }
                        with _StaticServer(snapshot.root) as server:
                            target_url = f"{server.url}/{source.entrypoint}"
                            for viewport in source.viewports:
                                context = browser.new_context(
                                    locale=str(protocol["locale"]),
                                    timezone_id=str(protocol["timezone"]),
                                    color_scheme=str(protocol["color_scheme"]),
                                    reduced_motion=str(protocol["reduced_motion"]),
                                    device_scale_factor=float(protocol["device_scale_factor"]),
                                )
                                blocked_requests: list[str] = []
                                context.route("**/*", _local_only_route(blocked_requests))
                                try:
                                    page = context.new_page()
                                    record = _capture_snapshot(
                                        page,
                                        source,
                                        viewport,
                                        target_url,
                                        artifact_root,
                                        origin,
                                        timeout_ms,
                                    )
                                    if blocked_requests:
                                        raise RuntimeError(
                                            f"{source.id} at {viewport} retained an unfrozen external request: "
                                            f"{blocked_requests[0]!r}"
                                        )
                                    records.append(record)
                                finally:
                                    context.close()
                except (OSError, RuntimeError, ValueError, PlaywrightError) as exc:
                    if not continue_on_error:
                        raise
                    failures.extend(
                        {"source_id": source.id, "viewport": viewport, "error": str(exc)}
                        for viewport in source.viewports
                    )
        finally:
            browser.close()
    vendor_payload = {
        "schema": VENDOR_RECEIPT_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "records": vendor_records,
    }
    return {
        "schema": CAPTURE_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "capture_protocol": dict(protocol),
        "runtime": {
            "browser_engine": "chromium",
            "browser_version": protocol["browser_version"],
            "playwright_version": importlib.metadata.version("playwright"),
        },
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "feature_extractor_script_sha256": hashlib.sha256(
            _computed_feature_script().encode("utf-8")
        ).hexdigest(),
        "captured_at": datetime.now(UTC).isoformat(),
        "source_receipts": source_receipts,
        "source_receipts_sha256": canonical_json_sha256(source_receipts),
        "vendor_receipts": vendor_payload,
        "vendor_receipts_sha256": canonical_json_sha256(vendor_payload),
        "records": records,
        "failures": failures,
        "status": "complete" if not failures else "partial_candidate_snapshot_capture",
        "note": (
            "Pinned source checkouts were rendered from a local overlay with frozen HTTPS visual assets and a "
            "recorded nonvisual analytics-script exclusion policy. This is an artifact-level snapshot of the "
            "declared source, not proof of individual authorship or deployment-to-commit parity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--checkout-receipts", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--vendor-timeout-s", type=int, default=ASSET_TIMEOUT_S)
    parser.add_argument("--max-asset-bytes", type=int, default=MAX_ASSET_BYTES)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--allow-partial-receipts",
        action="store_true",
        help="candidate-only: record missing source materializations as capture failures",
    )
    args = parser.parse_args()
    payload = capture_vendored_static_snapshots(
        _load_object(args.manifest),
        checkout_root=args.checkout_root,
        checkout_receipts_path=args.checkout_receipts,
        artifact_root=args.artifact_root,
        chrome=args.chrome,
        timeout_ms=args.timeout_ms,
        vendor_timeout_s=args.vendor_timeout_s,
        max_asset_bytes=args.max_asset_bytes,
        continue_on_error=args.continue_on_error,
        allow_partial_receipts=args.allow_partial_receipts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"captured {len(payload['records'])} vendored snapshot viewport records; failures: {len(payload['failures'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
