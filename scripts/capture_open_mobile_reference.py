#!/usr/bin/env python3
"""Capture computed CSS features for a validated open mobile reference manifest.

This is a corpus-construction utility, not an empirical result generator. It
renders every pinned source at its declared viewports and emits feature rows
that can later be split into a WebHumanBench train/dev/test manifest.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import importlib.metadata
import json
import re
import subprocess
import sys
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.open_reference import OpenReferenceSource, validate_open_reference_manifest
from webmark.pinned_build import (
    BUILD_CAPTURE_METHOD,
    BUILD_RECEIPT_SCHEMA,
    source_tree_sha256,
    validate_build_receipts,
)
from webmark.release import canonical_json_sha256, sha256_file
from webmark.vendor_snapshot import SNAPSHOT_CAPTURE_METHOD

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
CAPTURE_SCHEMA = "webmark_open_mobile_capture_v2"
FEATURE_EXTRACTOR_VERSION = "computed-style-v3"
CHECKOUT_RECEIPT_SCHEMAS = frozenset({
    "webmark_pinned_static_checkout_receipts_v1",
    "webmark_pinned_static_checkout_receipts_v2",
    "webmark_pinned_source_receipts_v3",
})
STATIC_CAPTURE_METHOD = "pinned_local_static_checkout"
STATIC_CAPTURE_METHODS = frozenset({STATIC_CAPTURE_METHOD, SNAPSHOT_CAPTURE_METHOD})
ENTRYPOINT_CLOSURE_MATERIALIZATION_METHOD = "github_entrypoint_closure_v1"


def _viewport(value: str) -> dict[str, int]:
    try:
        width, height = value.lower().split("x", 1)
        parsed = {"width": int(width), "height": int(height)}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid viewport {value!r}") from exc
    if parsed["width"] <= 0 or parsed["height"] <= 0:
        raise ValueError(f"invalid viewport {value!r}")
    return parsed


def _computed_feature_script() -> str:
    return """
    () => {
      const visible = (el) => {
        const box = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return box.width > 0 && box.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
      };
      const hasOwnText = (el) => Array.from(el.childNodes).some(
        (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim().length > 0
      );
      const parseRgb = (value) => {
        const match = value.match(/rgba?\\(([^)]+)\\)/);
        if (!match) return null;
        const parts = match[1].split(',').map((part) => parseFloat(part.trim()));
        if (parts.length < 3 || (parts.length > 3 && parts[3] === 0)) return null;
        return parts.slice(0, 3).map((part) => Math.max(0, Math.min(255, Math.round(part))));
      };
      const rgbString = (rgb) => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
      const saturation = (rgb) => {
        const values = rgb.map((value) => value / 255);
        const high = Math.max(...values);
        const low = Math.min(...values);
        return high === 0 ? 0 : (high - low) / high;
      };
      // Chromium preserves the CSS keyword `normal` in computed line-height.
      // v3 records the CSS initial-value multiplier (1.2) for that keyword so
      // pages without an explicit declaration remain measurable.
      const normalLineHeightRatio = 1.2;
      const typography = [];
      const spacing = [];
      const grid = [];
      const color = [];
      const saturationValues = [];
      for (const el of Array.from(document.querySelectorAll('body *'))) {
        if (!visible(el)) continue;
        const style = window.getComputedStyle(el);
        const box = el.getBoundingClientRect();
        if (hasOwnText(el)) {
          const fontSize = parseFloat(style.fontSize);
          const lineHeight = parseFloat(style.lineHeight);
          if (Number.isFinite(fontSize) && fontSize > 0) typography.push(fontSize);
          if (Number.isFinite(lineHeight) && lineHeight > 0 && Number.isFinite(fontSize) && fontSize > 0) {
            spacing.push(lineHeight / fontSize);
          } else if (style.lineHeight.trim().toLowerCase() === 'normal' && Number.isFinite(fontSize) && fontSize > 0) {
            spacing.push(normalLineHeightRatio);
          }
        }
        if (box.width > 0) grid.push(Number(((box.left / 8) % 12).toFixed(3)));
        for (const value of [style.color, style.backgroundColor]) {
          const rgb = parseRgb(value);
          if (!rgb) continue;
          color.push(rgbString(rgb));
          saturationValues.push(saturation(rgb));
        }
      }
      return { typography, spacing, grid, color, saturation: saturationValues };
    }
    """


def _capture_protocol(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the browser settings that materially affect computed styles."""
    declared = metadata.get("capture_protocol")
    if declared is None:
        if metadata.get("release_status") == "public":
            raise ValueError("a public source manifest requires metadata.capture_protocol")
        return {
            "browser_engine": "chromium",
            "browser_version": None,
            "locale": "en-US",
            "timezone": "UTC",
            "color_scheme": "light",
            "reduced_motion": "reduce",
            "device_scale_factor": 1,
        }
    if not isinstance(declared, Mapping):
        raise ValueError("metadata.capture_protocol must be an object")
    required = (
        "browser_engine",
        "browser_version",
        "locale",
        "timezone",
        "color_scheme",
        "reduced_motion",
        "device_scale_factor",
    )
    missing = [field for field in required if declared.get(field) in (None, "")]
    if missing:
        raise ValueError(f"metadata.capture_protocol is missing: {', '.join(missing)}")
    if str(declared["browser_engine"]).lower() != "chromium":
        raise ValueError("capture_open_mobile_reference.py supports only capture_protocol.browser_engine='chromium'")
    if declared["color_scheme"] not in {"light", "dark", "no-preference"}:
        raise ValueError("capture_protocol.color_scheme must be light, dark, or no-preference")
    if declared["reduced_motion"] not in {"reduce", "no-preference"}:
        raise ValueError("capture_protocol.reduced_motion must be reduce or no-preference")
    scale = declared["device_scale_factor"]
    if not isinstance(scale, int | float) or scale <= 0:
        raise ValueError("capture_protocol.device_scale_factor must be positive")
    return dict(declared)


def _artifact_stem(source_id: str, viewport: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("._") or "source"
    suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}__{suffix}__{viewport.replace('x', '_')}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_output(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git verification failed for {checkout}: {result.stderr.strip() or f'exit status {result.returncode}'}"
        )
    return result.stdout


def _load_static_checkout_receipts(
    path: Path,
    manifest: Mapping[str, Any],
    sources: list[OpenReferenceSource],
    checkout_root: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") not in CHECKOUT_RECEIPT_SCHEMAS:
        raise ValueError(f"checkout receipts must use one of {sorted(CHECKOUT_RECEIPT_SCHEMAS)!r}")
    if payload.get("source_manifest_sha256") != canonical_json_sha256(manifest):
        raise ValueError("checkout receipts source_manifest_sha256 does not match the source manifest")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkout receipts requires a non-empty records list")
    receipt_by_id: dict[str, Mapping[str, Any]] = {}
    local_sources = [source for source in sources if source.capture_method in STATIC_CAPTURE_METHODS]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("checkout receipt rows must be objects")
        source_id = str(row.get("source_id", ""))
        if not source_id or source_id in receipt_by_id:
            raise ValueError("checkout receipt source_id values must be non-empty and unique")
        receipt_by_id[source_id] = row
    expected_ids = {source.id for source in local_sources}
    if not allow_partial and set(receipt_by_id) != expected_ids:
        missing = sorted(expected_ids.difference(receipt_by_id))
        extra = sorted(set(receipt_by_id).difference(expected_ids))
        raise ValueError(f"checkout receipts do not match local sources; missing={missing[:3]}, extra={extra[:3]}")
    if allow_partial and not set(receipt_by_id).issubset(expected_ids):
        extra = sorted(set(receipt_by_id).difference(expected_ids))
        raise ValueError(f"checkout receipts contain unknown local sources: {extra[:3]}")

    for source in local_sources:
        if source.id not in receipt_by_id:
            if allow_partial:
                continue
            raise RuntimeError(f"checkout receipts are missing source {source.id!r}")
        receipt = receipt_by_id[source.id]
        if receipt.get("commit_sha") != source.commit_sha or receipt.get("entrypoint") != source.entrypoint:
            raise ValueError(f"checkout receipt does not match pinned source {source.id!r}")
        checkout_name = str(receipt.get("checkout_path", ""))
        checkout = checkout_root / checkout_name
        if checkout_name != source.id or checkout.resolve().parent != checkout_root.resolve() or not checkout.is_dir():
            raise ValueError(f"checkout receipt has an unsafe or missing checkout path for {source.id!r}")
        materialization_method = receipt.get("materialization_method", "git_checkout")
        if materialization_method == "git_checkout":
            head = _git_output(checkout, "rev-parse", "HEAD").strip()
            if head != source.commit_sha:
                raise ValueError(f"checkout {source.id!r} is not at its declared pinned commit")
            status = _git_output(checkout, "status", "--porcelain")
            if status.strip():
                raise ValueError(f"checkout {source.id!r} is dirty")
            tree = _git_output(checkout, "ls-tree", "-r", "--full-tree", source.commit_sha)
            if receipt.get("checkout_tree_sha256") != _sha256_text(tree):
                raise ValueError(f"checkout receipt tree digest does not match {source.id!r}")
        elif materialization_method == "github_api_tarball":
            tree_sha = str(receipt.get("github_tree_sha", ""))
            if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
                raise ValueError(f"GitHub API receipt needs a pinned tree SHA for {source.id!r}")
            archive_sha = str(receipt.get("source_archive_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
                raise ValueError(f"GitHub API receipt needs a source archive digest for {source.id!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("checkout_tree_sha256", ""))):
                raise ValueError(f"GitHub API receipt needs a source tree digest for {source.id!r}")
        elif materialization_method == "gitea_api_tarball":
            tree_ref = str(receipt.get("gitea_tree_ref", ""))
            if not re.fullmatch(r"[0-9a-f]{40}", tree_ref):
                raise ValueError(f"Gitea API receipt needs a pinned tree reference for {source.id!r}")
            archive_sha = str(receipt.get("source_archive_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
                raise ValueError(f"Gitea API receipt needs a source archive digest for {source.id!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("checkout_tree_sha256", ""))):
                raise ValueError(f"Gitea API receipt needs a source tree digest for {source.id!r}")
        elif materialization_method == ENTRYPOINT_CLOSURE_MATERIALIZATION_METHOD:
            if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("github_tree_sha", ""))):
                raise ValueError(f"entrypoint closure receipt needs a pinned GitHub tree SHA for {source.id!r}")
            if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("entrypoint_git_blob_sha1", ""))):
                raise ValueError(f"entrypoint closure receipt needs a pinned entrypoint blob SHA for {source.id!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("closure_manifest_sha256", ""))):
                raise ValueError(f"entrypoint closure receipt needs a closure manifest digest for {source.id!r}")
            if not isinstance(receipt.get("closure_file_count"), int) or int(receipt["closure_file_count"]) <= 0:
                raise ValueError(f"entrypoint closure receipt needs a positive closure file count for {source.id!r}")
        else:
            raise ValueError(f"unsupported checkout materialization method for {source.id!r}")
        entrypoint = checkout / source.entrypoint
        if not entrypoint.is_file() or receipt.get("entrypoint_sha256") != sha256_file(entrypoint):
            raise ValueError(f"checkout receipt entrypoint digest does not match {source.id!r}")
        if payload.get("schema") == "webmark_pinned_source_receipts_v3":
            if receipt.get("checkout_file_tree_sha256") != source_tree_sha256(checkout):
                raise ValueError(f"checkout receipt file-tree digest does not match {source.id!r}")
    return receipt_by_id


def _load_build_receipts(
    path: Path,
    manifest: Mapping[str, Any],
    sources: list[OpenReferenceSource],
    artifact_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    """Load build outputs that already passed the shared receipt contract."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("build receipts must contain a JSON object")
    if payload.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise ValueError(f"build receipts must use schema {BUILD_RECEIPT_SCHEMA!r}")
    records = validate_build_receipts(payload, manifest, sources, artifact_root=artifact_root)
    return records, dict(payload)


class _QuietStaticRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _StaticCheckoutServer:
    """Serve one fixed checkout on a loopback-only ephemeral port."""

    def __init__(self, directory: Path) -> None:
        handler = functools.partial(_QuietStaticRequestHandler, directory=str(directory))
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> _StaticCheckoutServer:
        self._thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _local_capture_target(server_url: str, entrypoint: str) -> str:
    return f"{server_url}/{quote(entrypoint)}"


def _local_only_route(blocked_requests: list[str]) -> Any:
    def handle(route: Any) -> None:
        request_url = route.request.url
        parsed = urlparse(request_url)
        if parsed.scheme in {"http", "https"} and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            blocked_requests.append(request_url)
            route.abort()
        else:
            route.continue_()

    return handle


def _capture(
    page: Any,
    source: OpenReferenceSource,
    viewport: str,
    timeout_ms: int,
    *,
    artifact_root: Path,
    target_url: str | None = None,
    capture_origin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    page.set_viewport_size(_viewport(viewport))
    response = page.goto(target_url or source.capture_url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(500)
    features = page.evaluate(_computed_feature_script())
    if not all(features.get(name) for name in ("typography", "spacing", "grid", "color", "saturation")):
        raise RuntimeError(f"{source.id} at {viewport} produced an incomplete computed feature vector")
    final_url = page.url
    parsed_final_url = urlparse(final_url)
    origin = dict(capture_origin or {"mode": "remote_observation"})
    if origin["mode"] in {STATIC_CAPTURE_METHOD, BUILD_CAPTURE_METHOD}:
        if parsed_final_url.scheme != "http" or parsed_final_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(f"{source.id} at {viewport} escaped its pinned local checkout: {final_url!r}")
    elif parsed_final_url.scheme != "https" or not parsed_final_url.netloc or parsed_final_url.username:
        raise RuntimeError(f"{source.id} at {viewport} redirected to unsupported final URL {final_url!r}")
    artifact_stem = _artifact_stem(source.id, viewport)
    html_path = artifact_root / "html" / f"{artifact_stem}.html"
    screenshot_path = artifact_root / "screenshots" / f"{artifact_stem}.png"
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
        "final_url": final_url,
        "http_status": response.status if response is not None else None,
        "capture_origin": origin,
        "captured_at": datetime.now(UTC).isoformat(),
        "features": features,
        "feature_sha256": canonical_json_sha256(features),
        "capture_html_sha256": sha256_file(html_path),
        "artifacts": {
            "html": {
                "path": html_path.relative_to(artifact_root).as_posix(),
                "sha256": sha256_file(html_path),
            },
            "screenshot": {
                "path": screenshot_path.relative_to(artifact_root).as_posix(),
                "sha256": sha256_file(screenshot_path),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="validated open-reference JSON manifest")
    parser.add_argument("--output", type=Path, required=True, help="captured feature rows JSON")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="directory for immutable captured HTML and PNG artifacts (default: <output stem>_artifacts)",
    )
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record candidate-capture failures and continue; never use this partial ledger as a public release",
    )
    parser.add_argument(
        "--allow-blocked-external",
        action="store_true",
        help="retain local-only candidate captures after blocked external requests; inspect and declare them before release",
    )
    parser.add_argument(
        "--allow-partial-receipts",
        action="store_true",
        help="candidate-only: record missing local source materializations as capture failures",
    )
    parser.add_argument(
        "--checkout-root",
        type=Path,
        help="root containing <source-id> detached checkouts for pinned_local_static_checkout sources",
    )
    parser.add_argument(
        "--checkout-receipts",
        type=Path,
        help="receipt emitted by materialize_pinned_static_sources.py for local static capture",
    )
    parser.add_argument(
        "--build-receipts",
        type=Path,
        help="receipt emitted by build_pinned_local_sources.py for pinned_local_build capture",
    )
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    sources = validate_open_reference_manifest(manifest)
    metadata = manifest["metadata"]
    if not isinstance(metadata, Mapping):  # Guarded by validate_open_reference_manifest.
        raise ValueError("manifest metadata must be an object")
    capture_protocol = _capture_protocol(metadata)
    if args.allow_partial_receipts and (
        not args.continue_on_error or metadata.get("release_status") == "public"
    ):
        raise ValueError(
            "--allow-partial-receipts requires --continue-on-error and a non-public candidate manifest"
        )
    artifact_root = args.artifact_root or args.output.with_name(f"{args.output.stem}_artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    local_sources = [source for source in sources if source.capture_method in STATIC_CAPTURE_METHODS]
    build_sources = [source for source in sources if source.capture_method == BUILD_CAPTURE_METHOD]
    unsupported_local = [
        source.id
        for source in sources
        if source.capture_method not in {"remote_observation", STATIC_CAPTURE_METHOD, BUILD_CAPTURE_METHOD}
    ]
    if unsupported_local:
        raise ValueError(
            "capture_open_mobile_reference.py currently supports only remote observations, "
            f"{STATIC_CAPTURE_METHOD}, and {BUILD_CAPTURE_METHOD}; unsupported examples: "
            f"{', '.join(unsupported_local[:3])}"
        )
    if local_sources and (args.checkout_root is None or args.checkout_receipts is None):
        raise ValueError("pinned local static sources require both --checkout-root and --checkout-receipts")
    receipts: dict[str, Mapping[str, Any]] = {}
    if local_sources:
        assert args.checkout_root is not None and args.checkout_receipts is not None
        receipts = _load_static_checkout_receipts(
            args.checkout_receipts,
            manifest,
            sources,
            args.checkout_root,
            allow_partial=args.allow_partial_receipts,
        )
    build_receipts: dict[str, Mapping[str, Any]] = {}
    build_receipt_payload: dict[str, Any] | None = None
    if build_sources:
        if args.build_receipts is None:
            raise ValueError("pinned local build sources require --build-receipts")
        build_receipts, build_receipt_payload = _load_build_receipts(
            args.build_receipts, manifest, sources, artifact_root
        )
    try:
        import playwright.sync_api as playwright_sync_api
    except ImportError as exc:  # pragma: no cover - optional browser dependency
        raise SystemExit('Install browser dependencies with: pip install -e ".[browser]"') from exc

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with playwright_sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(args.chrome) if args.chrome.exists() else None,
        )
        try:
            browser_version = browser.version
            expected_version = capture_protocol["browser_version"]
            if expected_version is not None and expected_version != browser_version:
                raise RuntimeError(
                    "capture_protocol.browser_version does not match the launched browser: "
                    f"expected {expected_version!r}, got {browser_version!r}"
                )
            runtime_protocol = {
                **capture_protocol,
                "browser_engine": "chromium",
                "browser_version": browser_version,
            }
            for source in sources:
                receipt = receipts.get(source.id)
                build_receipt = build_receipts.get(source.id)
                if source.capture_method in STATIC_CAPTURE_METHODS and receipt is None:
                    message = f"{source.id} was not materialized into the candidate receipt"
                    if not args.continue_on_error:
                        raise RuntimeError(message)
                    failures.extend(
                        {"source_id": source.id, "viewport": viewport, "error": message}
                        for viewport in source.viewports
                    )
                    continue
                if receipt is None and build_receipt is None:
                    capture_url = source.capture_url
                    capture_origin: Mapping[str, Any] = {"mode": "remote_observation"}
                    static_server: _StaticCheckoutServer | None = None
                elif receipt is not None:
                    assert args.checkout_root is not None
                    checkout = args.checkout_root / str(receipt["checkout_path"])
                    static_server = _StaticCheckoutServer(checkout)
                    static_server.__enter__()
                    capture_url = _local_capture_target(static_server.url, source.entrypoint)
                    capture_origin = {
                        "mode": source.capture_method,
                        "commit_sha": source.commit_sha,
                        "checkout_tree_sha256": receipt["checkout_tree_sha256"],
                        "entrypoint_sha256": receipt["entrypoint_sha256"],
                    }
                else:
                    assert build_receipt is not None and source.build_recipe is not None
                    output = artifact_root / str(build_receipt["output_directory"])
                    static_server = _StaticCheckoutServer(output)
                    static_server.__enter__()
                    capture_url = _local_capture_target(static_server.url, source.build_recipe.output_entrypoint)
                    capture_origin = {
                        "mode": BUILD_CAPTURE_METHOD,
                        "commit_sha": source.commit_sha,
                        "checkout_tree_sha256": build_receipt["checkout_tree_sha256"],
                        "entrypoint_sha256": build_receipt["entrypoint_sha256"],
                        "build_receipt_sha256": canonical_json_sha256(build_receipt),
                        "build_recipe_sha256": build_receipt["build_recipe_sha256"],
                        "output_tree_sha256": build_receipt["output_tree_sha256"],
                        "output_entrypoint_sha256": build_receipt["output_entrypoint_sha256"],
                    }
                try:
                    for viewport in source.viewports:
                        # A fresh browser context prevents cookies, local storage,
                        # service workers, and viewport state from crossing rows.
                        context = browser.new_context(
                            locale=str(runtime_protocol["locale"]),
                            timezone_id=str(runtime_protocol["timezone"]),
                            color_scheme=str(runtime_protocol["color_scheme"]),
                            reduced_motion=str(runtime_protocol["reduced_motion"]),
                            device_scale_factor=float(runtime_protocol["device_scale_factor"]),
                        )
                        blocked_requests: list[str] = []
                        if receipt is not None or build_receipt is not None:
                            context.route("**/*", _local_only_route(blocked_requests))
                        try:
                            page = context.new_page()
                            record = _capture(
                                page,
                                source,
                                viewport,
                                args.timeout_ms,
                                artifact_root=artifact_root,
                                target_url=capture_url,
                                capture_origin=capture_origin,
                            )
                            if blocked_requests and not args.allow_blocked_external:
                                raise RuntimeError(
                                    f"{source.id} at {viewport} requested resources outside the pinned checkout: "
                                    f"{blocked_requests[0]!r}"
                                )
                            if blocked_requests:
                                record["blocked_external_requests"] = blocked_requests
                            records.append(record)
                        except (OSError, playwright_sync_api.Error, RuntimeError, ValueError) as exc:
                            if not args.continue_on_error:
                                raise
                            failures.append({
                                "source_id": source.id,
                                "viewport": viewport,
                                "error": str(exc),
                            })
                        finally:
                            context.close()
                finally:
                    if static_server is not None:
                        static_server.__exit__(None, None, None)
        finally:
            browser.close()

    payload = {
        "schema": CAPTURE_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "capture_protocol": runtime_protocol,
        "runtime": {
            "browser_engine": "chromium",
            "browser_version": runtime_protocol["browser_version"],
            "playwright_version": importlib.metadata.version("playwright"),
        },
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "feature_extractor_script_sha256": hashlib.sha256(
            _computed_feature_script().encode("utf-8")
        ).hexdigest(),
        "captured_at": datetime.now(UTC).isoformat(),
        "records": records,
        "failures": failures,
        "status": "complete" if not failures else "partial_candidate_probe",
        "note": (
            "Computed-style capture ledger with immutable HTML and screenshot digests. "
            "Remote observations remain candidate-only; public sources require a pinned local capture origin. "
            "A partial candidate probe must be filtered and recaptured without --continue-on-error before release. "
            "Blocked external requests require explicit source-level review before a public release."
        ),
    }
    if build_receipt_payload is not None:
        payload["build_receipts"] = build_receipt_payload
        payload["build_receipts_sha256"] = canonical_json_sha256(build_receipt_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"captured {len(records)} viewport records from {len(sources)} open sources; failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
