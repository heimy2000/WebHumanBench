"""Cross-artifact checks required before publishing a WebHumanBench corpus.

The protocol validator checks individual manifests. This module checks the
release boundary: every scored human row must point to a pinned source and an
immutable local capture, and every generated row must retain generation
provenance. It deliberately validates record completeness rather than claiming
that a URL proves human authorship or that a score measures preference.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .human_likeness import BenchmarkRecord, record_from_dict, validate_manifest
from .open_reference import (
    HISTORICAL_EVIDENCE_SCHEMA,
    PRIMARY_MOBILE_VIEWPORTS,
    OpenReferenceSource,
    validate_open_reference_manifest,
)
from .pinned_build import BUILD_CAPTURE_METHOD, BUILD_RECEIPT_SCHEMA, validate_build_receipts
from .vendor_snapshot import (
    SNAPSHOT_CAPTURE_METHOD,
    VENDOR_RECEIPT_SCHEMA,
    validate_vendor_receipts,
)

CAPTURE_SCHEMA = "webmark_open_mobile_capture_v2"
PUBLIC_RELEASE_SCHEMA = "webmark_public_release_audit_v1"
SOURCE_RECEIPT_SCHEMA = "webmark_pinned_source_receipts_v3"
ENTRYPOINT_CLOSURE_MATERIALIZATION_METHOD = "github_entrypoint_closure_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{7,64}")


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON semantically so whitespace does not change provenance links."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a release artifact without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _sha256(value: Any, field: str) -> str:
    text = _nonempty_string(value, field)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _revision(value: Any, field: str) -> str:
    text = _nonempty_string(value, field)
    if not REVISION_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 7-64 character hexadecimal revision")
    return text


def _timestamp(value: Any, field: str) -> None:
    text = _nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    parsed.astimezone(UTC)


def _https_url(value: Any, field: str) -> None:
    text = _nonempty_string(value, field)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field} must be an HTTPS URL")


def _relative_artifact_path(value: Any, field: str) -> PurePosixPath:
    text = _nonempty_string(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError(f"{field} must be a portable relative path")
    return path


def _check_capture_protocol(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = _mapping(metadata.get("capture_protocol"), "source metadata capture_protocol")
    for field in (
        "browser_engine",
        "browser_version",
        "locale",
        "timezone",
        "color_scheme",
        "reduced_motion",
    ):
        _nonempty_string(protocol.get(field), f"source metadata capture_protocol.{field}")
    scale = protocol.get("device_scale_factor")
    if not isinstance(scale, int | float) or scale <= 0:
        raise ValueError("source metadata capture_protocol.device_scale_factor must be positive")
    return protocol


def _validate_public_source_metadata(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    metadata = _mapping(manifest.get("metadata"), "source metadata")
    if metadata.get("release_status") != "public":
        raise ValueError("source metadata release_status must be 'public'")
    _revision(metadata.get("release_revision"), "source metadata release_revision")
    _nonempty_string(metadata.get("data_license"), "source metadata data_license")
    protocol = _check_capture_protocol(metadata)
    if manifest.get("schema") == HISTORICAL_EVIDENCE_SCHEMA:
        policy = _mapping(metadata.get("provenance_policy"), "source metadata provenance_policy")
        if policy.get("mode") != "historical_open_source_evidence_v1":
            raise ValueError(
                "source metadata provenance_policy.mode must be 'historical_open_source_evidence_v1'"
            )
        minimum_evidence = policy.get("min_distinct_evidence_kinds")
        if not isinstance(minimum_evidence, int) or minimum_evidence < 2:
            raise ValueError(
                "source metadata provenance_policy.min_distinct_evidence_kinds must be at least 2"
            )
        return protocol, {"mode": "historical_open_source_evidence", "minimum": minimum_evidence}
    policy = _mapping(metadata.get("curation_policy"), "source metadata curation_policy")
    minimum_reviews = policy.get("min_independent_reviews")
    if not isinstance(minimum_reviews, int) or minimum_reviews < 2:
        raise ValueError(
            "source metadata curation_policy.min_independent_reviews must be at least 2"
        )
    return protocol, {"mode": "independent_reviews", "minimum": minimum_reviews}


def _validate_provenance_reviews(raw_source: Mapping[str, Any], minimum_reviews: int) -> None:
    source_id = _nonempty_string(raw_source.get("id"), "source id")
    reviews = raw_source.get("provenance_reviews")
    if not isinstance(reviews, list) or len(reviews) < minimum_reviews:
        raise ValueError(
            f"source {source_id!r} requires at least {minimum_reviews} independent provenance reviews"
        )
    reviewer_ids: list[str] = []
    for index, review in enumerate(reviews):
        item = _mapping(review, f"source {source_id!r} provenance_reviews[{index}]")
        reviewer_id = _nonempty_string(
            item.get("reviewer_id"), f"source {source_id!r} provenance_reviews[{index}].reviewer_id"
        )
        reviewer_ids.append(reviewer_id)
        _timestamp(
            item.get("reviewed_at"), f"source {source_id!r} provenance_reviews[{index}].reviewed_at"
        )
        if item.get("decision") != "admit":
            raise ValueError(
                f"source {source_id!r} provenance review {index} must have decision 'admit'"
            )
        evidence_urls = item.get("evidence_urls")
        if not isinstance(evidence_urls, list) or not evidence_urls:
            raise ValueError(
                f"source {source_id!r} provenance review {index} requires evidence_urls"
            )
        for evidence_index, evidence_url in enumerate(evidence_urls):
            _https_url(
                evidence_url,
                f"source {source_id!r} provenance_reviews[{index}].evidence_urls[{evidence_index}]",
            )
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ValueError(f"source {source_id!r} provenance review IDs must be distinct")


def _validate_historical_provenance_evidence(
    raw_source: Mapping[str, Any], minimum_kinds: int
) -> None:
    """Recheck the proxy-label evidence required by the public release boundary."""
    source_id = _nonempty_string(raw_source.get("id"), "source id")
    commit_sha = _revision(raw_source.get("commit_sha"), f"source {source_id!r} commit_sha")
    evidence = raw_source.get("provenance_evidence")
    if not isinstance(evidence, list) or len(evidence) < minimum_kinds:
        raise ValueError(
            f"source {source_id!r} requires at least {minimum_kinds} historical provenance evidence entries"
        )
    kinds: set[str] = set()
    pinned_history = False
    for index, raw_item in enumerate(evidence):
        item = _mapping(raw_item, f"source {source_id!r} provenance_evidence[{index}]")
        kind = _nonempty_string(
            item.get("kind"), f"source {source_id!r} provenance_evidence[{index}].kind"
        )
        if kind not in {"pinned_git_history", "source_project_identity"}:
            raise ValueError(
                f"source {source_id!r} provenance_evidence[{index}].kind is unsupported: {kind!r}"
            )
        kinds.add(kind)
        urls = item.get("evidence_urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError(
                f"source {source_id!r} provenance_evidence[{index}] requires evidence_urls"
            )
        for url_index, url in enumerate(urls):
            _https_url(
                url, f"source {source_id!r} provenance_evidence[{index}].evidence_urls[{url_index}]"
            )
            parsed = urlparse(str(url))
            if kind == "pinned_git_history" and (
                commit_sha in parsed.path.split("/") or commit_sha in parsed.query
            ):
                pinned_history = True
    if len(kinds) < minimum_kinds:
        raise ValueError(
            f"source {source_id!r} historical provenance evidence requires {minimum_kinds} distinct kinds"
        )
    if not pinned_history:
        raise ValueError(
            f"source {source_id!r} needs pinned_git_history evidence for its commit_sha"
        )


def _validate_capture_artifact(
    artifact: Any,
    field: str,
    artifact_root: Path | None,
) -> str:
    item = _mapping(artifact, field)
    relative_path = _relative_artifact_path(item.get("path"), f"{field}.path")
    digest = _sha256(item.get("sha256"), f"{field}.sha256")
    if artifact_root is not None:
        path = artifact_root / relative_path
        if not path.is_file():
            raise ValueError(f"{field}.path does not exist under artifact_root: {relative_path}")
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(f"{field}.sha256 does not match {relative_path}")
    return digest


def _validate_capture_origin(
    record: Mapping[str, Any], source: OpenReferenceSource, final_url: str, capture_id: str
) -> None:
    """Require public captures to identify the source revision actually served.

    A public source manifest may retain an HTTPS canonical URL for citation,
    but its scored capture must be served from a fixed local checkout. Remote
    deployments can drift after the pinned commit and are therefore useful only
    as candidate observations, never as public reference evidence.
    """
    origin = _mapping(record.get("capture_origin"), f"capture {capture_id!r} capture_origin")
    mode = _nonempty_string(origin.get("mode"), f"capture {capture_id!r} capture_origin.mode")
    if mode != source.capture_method:
        raise ValueError(
            f"capture {capture_id!r} capture_origin.mode does not match source {source.id!r} capture_method"
        )
    parsed_final_url = urlparse(final_url)
    if mode == "remote_observation":
        if (
            parsed_final_url.scheme != "https"
            or not parsed_final_url.netloc
            or parsed_final_url.username
            or parsed_final_url.password
        ):
            raise ValueError(
                f"capture {capture_id!r} remote final_url must be an HTTPS URL without credentials"
            )
        return
    if mode not in {"pinned_local_static_checkout", SNAPSHOT_CAPTURE_METHOD, "pinned_local_build"}:
        raise ValueError(f"capture {capture_id!r} has unsupported capture_origin.mode {mode!r}")
    if origin.get("commit_sha") != source.commit_sha:
        raise ValueError(
            f"capture {capture_id!r} capture_origin.commit_sha does not match its source"
        )
    _sha256(
        origin.get("checkout_tree_sha256"),
        f"capture {capture_id!r} capture_origin.checkout_tree_sha256",
    )
    _sha256(
        origin.get("entrypoint_sha256"), f"capture {capture_id!r} capture_origin.entrypoint_sha256"
    )
    if (
        parsed_final_url.scheme != "http"
        or parsed_final_url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed_final_url.username
        or parsed_final_url.password
    ):
        raise ValueError(
            f"capture {capture_id!r} pinned local final_url must be an HTTP localhost URL without credentials"
        )
    if mode == BUILD_CAPTURE_METHOD:
        for field in (
            "build_receipt_sha256",
            "build_recipe_sha256",
            "output_tree_sha256",
            "output_entrypoint_sha256",
        ):
            _sha256(origin.get(field), f"capture {capture_id!r} capture_origin.{field}")
    if mode == SNAPSHOT_CAPTURE_METHOD:
        _sha256(
            origin.get("vendor_receipt_sha256"),
            f"capture {capture_id!r} capture_origin.vendor_receipt_sha256",
        )
        _sha256(
            origin.get("snapshot_entrypoint_sha256"),
            f"capture {capture_id!r} capture_origin.snapshot_entrypoint_sha256",
        )


def _validate_snapshot_source_receipts(
    ledger: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    sources: list[OpenReferenceSource],
) -> dict[str, Mapping[str, Any]]:
    """Validate optional fixed-commit closure receipts retained with snapshots.

    Older public packages did not carry materialization receipts, so their
    absence remains supported. When a release includes them, every snapshot
    source and its capture origin must bind to the same receipt.
    """
    raw_receipts = ledger.get("source_receipts")
    raw_digest = ledger.get("source_receipts_sha256")
    if raw_receipts is None:
        if raw_digest is not None:
            raise ValueError("capture ledger has source_receipts_sha256 but no source_receipts")
        return {}
    payload = _mapping(raw_receipts, "capture ledger source_receipts")
    if payload.get("schema") != SOURCE_RECEIPT_SCHEMA:
        raise ValueError(f"capture ledger source_receipts must use {SOURCE_RECEIPT_SCHEMA!r}")
    if raw_digest != canonical_json_sha256(payload):
        raise ValueError("capture ledger source_receipts_sha256 does not match source_receipts")
    if payload.get("source_manifest_sha256") != canonical_json_sha256(source_manifest):
        raise ValueError("source receipts source_manifest_sha256 does not match source manifest")
    failures = payload.get("failures")
    if not isinstance(failures, list) or failures:
        raise ValueError("public snapshot source receipts require an empty failures list")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("source receipts requires a records list")

    expected = {
        source.id: source for source in sources if source.capture_method == SNAPSHOT_CAPTURE_METHOD
    }
    records: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"source receipt {index}")
        source_id = _nonempty_string(row.get("source_id"), f"source receipt {index} source_id")
        source = expected.get(source_id)
        if source is None or source_id in records:
            raise ValueError("source receipt IDs must identify one snapshot source each")
        if row.get("commit_sha") != source.commit_sha or row.get("entrypoint") != source.entrypoint:
            raise ValueError(f"source receipt {source_id!r} does not match its pinned source")
        if row.get("materialization_method") != ENTRYPOINT_CLOSURE_MATERIALIZATION_METHOD:
            raise ValueError(
                f"source receipt {source_id!r} has an unsupported materialization method"
            )
        for field in (
            "checkout_tree_sha256",
            "entrypoint_sha256",
            "closure_manifest_sha256",
        ):
            _sha256(row.get(field), f"source receipt {source_id!r} {field}")
        if not isinstance(row.get("closure_file_count"), int) or int(row["closure_file_count"]) < 1:
            raise ValueError(f"source receipt {source_id!r} requires a positive closure_file_count")
        records[source_id] = row
    if set(records) != set(expected):
        missing = sorted(set(expected).difference(records))
        extra = sorted(set(records).difference(expected))
        raise ValueError(
            f"source receipts do not match snapshot sources; missing={missing[:3]}, extra={extra[:3]}"
        )
    return records


def _validate_capture_ledger(
    ledger: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    sources: list[OpenReferenceSource],
    source_protocol: Mapping[str, Any],
    artifact_root: Path | None,
) -> dict[str, Mapping[str, Any]]:
    if ledger.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"capture ledger schema must be {CAPTURE_SCHEMA!r}")
    source_digest = canonical_json_sha256(source_manifest)
    if ledger.get("source_manifest_sha256") != source_digest:
        raise ValueError(
            "capture ledger source_manifest_sha256 does not match the supplied source manifest"
        )
    ledger_protocol = _mapping(ledger.get("capture_protocol"), "capture ledger capture_protocol")
    if dict(ledger_protocol) != dict(source_protocol):
        raise ValueError(
            "capture ledger capture_protocol does not match the public source manifest"
        )
    runtime = _mapping(ledger.get("runtime"), "capture ledger runtime")
    if runtime.get("browser_engine") != ledger_protocol.get("browser_engine"):
        raise ValueError("capture ledger runtime browser_engine does not match capture_protocol")
    if runtime.get("browser_version") != ledger_protocol.get("browser_version"):
        raise ValueError("capture ledger runtime browser_version does not match capture_protocol")
    _nonempty_string(runtime.get("playwright_version"), "capture ledger runtime playwright_version")
    _timestamp(ledger.get("captured_at"), "capture ledger captured_at")
    _nonempty_string(
        ledger.get("feature_extractor_version"), "capture ledger feature_extractor_version"
    )
    _sha256(
        ledger.get("feature_extractor_script_sha256"),
        "capture ledger feature_extractor_script_sha256",
    )

    build_sources = [source for source in sources if source.capture_method == BUILD_CAPTURE_METHOD]
    build_receipts: dict[str, Mapping[str, Any]] = {}
    if build_sources:
        raw_build_receipts = _mapping(ledger.get("build_receipts"), "capture ledger build_receipts")
        if raw_build_receipts.get("schema") != BUILD_RECEIPT_SCHEMA:
            raise ValueError(f"capture ledger build_receipts must use {BUILD_RECEIPT_SCHEMA!r}")
        if ledger.get("build_receipts_sha256") != canonical_json_sha256(raw_build_receipts):
            raise ValueError("capture ledger build_receipts_sha256 does not match build_receipts")
        build_receipts = validate_build_receipts(
            raw_build_receipts,
            source_manifest,
            sources,
            artifact_root=artifact_root,
        )
    elif (
        ledger.get("build_receipts") is not None or ledger.get("build_receipts_sha256") is not None
    ):
        raise ValueError(
            "capture ledger has build receipts but its source manifest has no pinned_local_build source"
        )

    snapshot_sources = [
        source for source in sources if source.capture_method == SNAPSHOT_CAPTURE_METHOD
    ]
    source_receipts = _validate_snapshot_source_receipts(ledger, source_manifest, sources)
    vendor_receipts: dict[str, Mapping[str, Any]] = {}
    if snapshot_sources:
        raw_vendor_receipts = _mapping(
            ledger.get("vendor_receipts"), "capture ledger vendor_receipts"
        )
        if raw_vendor_receipts.get("schema") != VENDOR_RECEIPT_SCHEMA:
            raise ValueError(f"capture ledger vendor_receipts must use {VENDOR_RECEIPT_SCHEMA!r}")
        if ledger.get("vendor_receipts_sha256") != canonical_json_sha256(raw_vendor_receipts):
            raise ValueError("capture ledger vendor_receipts_sha256 does not match vendor_receipts")
        vendor_receipts = validate_vendor_receipts(
            raw_vendor_receipts,
            source_manifest,
            sources,
            artifact_root=artifact_root,
        )
    elif (
        ledger.get("vendor_receipts") is not None
        or ledger.get("vendor_receipts_sha256") is not None
    ):
        raise ValueError(
            "capture ledger has vendor receipts but its source manifest has no static snapshot source"
        )

    raw_records = ledger.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("capture ledger requires a non-empty records list")
    source_by_id = {source.id: source for source in sources}
    captures: dict[str, Mapping[str, Any]] = {}
    capture_keys: set[tuple[str, str]] = set()
    for raw_record in raw_records:
        record = _mapping(raw_record, "capture ledger record")
        capture_id = _nonempty_string(record.get("id"), "capture ledger record id")
        if capture_id in captures:
            raise ValueError(f"capture ledger has duplicate capture id {capture_id!r}")
        source_id = _nonempty_string(record.get("source_id"), f"capture {capture_id!r} source_id")
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(
                f"capture {capture_id!r} references an unknown source_id {source_id!r}"
            )
        if record.get("source") != "human":
            raise ValueError(f"capture {capture_id!r} source must be 'human'")
        if record.get("group_id") != source.group_id:
            raise ValueError(f"capture {capture_id!r} group_id does not match source {source_id!r}")
        if record.get("page_type") != source.page_type:
            raise ValueError(
                f"capture {capture_id!r} page_type does not match source {source_id!r}"
            )
        viewport = _nonempty_string(record.get("viewport"), f"capture {capture_id!r} viewport")
        if viewport not in source.viewports:
            raise ValueError(
                f"capture {capture_id!r} viewport is not declared by source {source_id!r}"
            )
        capture_key = (source_id, viewport)
        if capture_key in capture_keys:
            raise ValueError(f"capture ledger has duplicate source/viewport pair {capture_key!r}")
        capture_keys.add(capture_key)
        if record.get("capture_url") != source.capture_url:
            raise ValueError(
                f"capture {capture_id!r} capture_url does not match source {source_id!r}"
            )
        final_url = _nonempty_string(record.get("final_url"), f"capture {capture_id!r} final_url")
        _validate_capture_origin(record, source, final_url, capture_id)
        if source.capture_method == BUILD_CAPTURE_METHOD:
            origin = _mapping(
                record.get("capture_origin"), f"capture {capture_id!r} capture_origin"
            )
            build_receipt = build_receipts.get(source.id)
            if build_receipt is None:
                raise ValueError(
                    f"capture {capture_id!r} has no build receipt for source {source.id!r}"
                )
            if origin.get("build_receipt_sha256") != canonical_json_sha256(build_receipt):
                raise ValueError(
                    f"capture {capture_id!r} build receipt digest does not match its ledger"
                )
            for field in ("build_recipe_sha256", "output_tree_sha256", "output_entrypoint_sha256"):
                if origin.get(field) != build_receipt.get(field):
                    raise ValueError(
                        f"capture {capture_id!r} capture origin {field} does not match its build receipt"
                    )
        if source.capture_method == SNAPSHOT_CAPTURE_METHOD:
            origin = _mapping(
                record.get("capture_origin"), f"capture {capture_id!r} capture_origin"
            )
            source_receipt = source_receipts.get(source.id)
            if source_receipt is not None:
                if origin.get("source_receipt_sha256") != canonical_json_sha256(source_receipt):
                    raise ValueError(
                        f"capture {capture_id!r} source receipt digest does not match its ledger"
                    )
                if origin.get("source_materialization_method") != source_receipt.get(
                    "materialization_method"
                ):
                    raise ValueError(
                        f"capture {capture_id!r} source materialization method does not match its receipt"
                    )
                if origin.get("closure_manifest_sha256") != source_receipt.get(
                    "closure_manifest_sha256"
                ):
                    raise ValueError(
                        f"capture {capture_id!r} closure manifest digest does not match its receipt"
                    )
            vendor_receipt = vendor_receipts.get(source.id)
            if vendor_receipt is None:
                raise ValueError(
                    f"capture {capture_id!r} has no vendored snapshot receipt for source {source.id!r}"
                )
            if origin.get("vendor_receipt_sha256") != canonical_json_sha256(vendor_receipt):
                raise ValueError(
                    f"capture {capture_id!r} vendor receipt digest does not match its ledger"
                )
            if origin.get("snapshot_entrypoint_sha256") != vendor_receipt.get(
                "snapshot_entrypoint_sha256"
            ):
                raise ValueError(
                    f"capture {capture_id!r} snapshot entrypoint digest does not match vendor receipt"
                )
        http_status = record.get("http_status")
        if not isinstance(http_status, int) or not 200 <= http_status < 400:
            raise ValueError(f"capture {capture_id!r} requires an HTTP status in [200, 400)")
        _timestamp(record.get("captured_at"), f"capture {capture_id!r} captured_at")

        # Reuse the benchmark feature parser without treating the capture as a scored row.
        record_from_dict(
            {
                "id": capture_id,
                "source": "human",
                "split": "test",
                "group_id": source.group_id,
                "page_type": source.page_type,
                "viewport": viewport,
                "features": record.get("features"),
            }
        )
        feature_digest = _sha256(
            record.get("feature_sha256"), f"capture {capture_id!r} feature_sha256"
        )
        if feature_digest != canonical_json_sha256(record.get("features")):
            raise ValueError(
                f"capture {capture_id!r} feature_sha256 does not match its feature payload"
            )
        html_digest = _sha256(
            record.get("capture_html_sha256"), f"capture {capture_id!r} capture_html_sha256"
        )
        artifacts = _mapping(record.get("artifacts"), f"capture {capture_id!r} artifacts")
        artifact_html = _validate_capture_artifact(
            artifacts.get("html"), f"capture {capture_id!r} artifacts.html", artifact_root
        )
        _validate_capture_artifact(
            artifacts.get("screenshot"),
            f"capture {capture_id!r} artifacts.screenshot",
            artifact_root,
        )
        if artifact_html != html_digest:
            raise ValueError(
                f"capture {capture_id!r} HTML artifact digest does not match capture_html_sha256"
            )
        captures[capture_id] = record

    for source in sources:
        for viewport in PRIMARY_MOBILE_VIEWPORTS:
            if (source.id, viewport) not in capture_keys:
                raise ValueError(
                    f"source {source.id!r} lacks required captured viewport {viewport!r}"
                )
    return captures


def _validate_public_benchmark_metadata(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    capture_ledger: Mapping[str, Any],
) -> str:
    metadata = _mapping(manifest.get("metadata"), "benchmark metadata")
    if metadata.get("release_status") != "public":
        raise ValueError("benchmark metadata release_status must be 'public'")
    if metadata.get("source_manifest_sha256") != canonical_json_sha256(source_manifest):
        raise ValueError(
            "benchmark metadata source_manifest_sha256 does not match the supplied source manifest"
        )
    if metadata.get("capture_ledger_sha256") != canonical_json_sha256(capture_ledger):
        raise ValueError(
            "benchmark metadata capture_ledger_sha256 does not match the supplied capture ledger"
        )
    _revision(metadata.get("release_revision"), "benchmark metadata release_revision")
    extractor = _nonempty_string(
        metadata.get("feature_extractor_version"), "benchmark metadata feature_extractor_version"
    )
    if metadata.get("scoring_unit") != "source_group":
        raise ValueError("benchmark metadata scoring_unit must be 'source_group'")
    viewport = _nonempty_string(
        metadata.get("scoring_viewport"), "benchmark metadata scoring_viewport"
    )
    if viewport not in PRIMARY_MOBILE_VIEWPORTS:
        raise ValueError(
            f"benchmark metadata scoring_viewport must be one of {list(PRIMARY_MOBILE_VIEWPORTS)!r}"
        )
    return extractor


def _validate_public_human_record(
    raw_record: Mapping[str, Any],
    record: BenchmarkRecord,
    source_by_id: Mapping[str, OpenReferenceSource],
    captures: Mapping[str, Mapping[str, Any]],
    scoring_viewport: str,
) -> None:
    provenance = _mapping(raw_record.get("provenance"), f"human record {record.id!r} provenance")
    source_id = _nonempty_string(
        provenance.get("source_id"), f"human record {record.id!r} provenance.source_id"
    )
    source = source_by_id.get(source_id)
    if source is None:
        raise ValueError(f"human record {record.id!r} references unknown source_id {source_id!r}")
    if record.group_id != source.group_id or record.page_type != source.page_type:
        raise ValueError(
            f"human record {record.id!r} does not match source {source_id!r} group or page type"
        )
    if record.leakage_group_id != source.group_id:
        raise ValueError(f"human record {record.id!r} leakage_group_id must match its source group")
    if record.viewport != scoring_viewport:
        raise ValueError(f"human record {record.id!r} must use the declared scoring_viewport")
    capture_id = _nonempty_string(
        provenance.get("capture_id"), f"human record {record.id!r} provenance.capture_id"
    )
    capture = captures.get(capture_id)
    if capture is None:
        raise ValueError(f"human record {record.id!r} references unknown capture_id {capture_id!r}")
    if capture.get("source_id") != source_id or capture.get("viewport") != record.viewport:
        raise ValueError(
            f"human record {record.id!r} capture does not match its source or viewport"
        )
    feature_digest = _sha256(
        provenance.get("feature_sha256"), f"human record {record.id!r} provenance.feature_sha256"
    )
    if feature_digest != canonical_json_sha256(raw_record.get("features")):
        raise ValueError(
            f"human record {record.id!r} feature_sha256 does not match its feature payload"
        )
    if feature_digest != capture.get("feature_sha256"):
        raise ValueError(f"human record {record.id!r} feature_sha256 does not match its capture")
    html_digest = _sha256(
        provenance.get("capture_html_sha256"),
        f"human record {record.id!r} provenance.capture_html_sha256",
    )
    if html_digest != capture.get("capture_html_sha256"):
        raise ValueError(
            f"human record {record.id!r} capture_html_sha256 does not match its capture"
        )


def _validate_public_ai_record(raw_record: Mapping[str, Any], record: BenchmarkRecord) -> None:
    provenance = _mapping(raw_record.get("provenance"), f"AI record {record.id!r} provenance")
    for field in ("prompt_id", "provider", "generated_at", "feature_extractor_version"):
        _nonempty_string(provenance.get(field), f"AI record {record.id!r} provenance.{field}")
    _timestamp(provenance.get("generated_at"), f"AI record {record.id!r} provenance.generated_at")
    for field in (
        "prompt_sha256",
        "generation_config_sha256",
        "raw_response_sha256",
        "generated_html_sha256",
        "rendered_html_sha256",
        "screenshot_sha256",
        "computed_feature_sha256",
    ):
        _sha256(provenance.get(field), f"AI record {record.id!r} provenance.{field}")
    if provenance.get("model_id") != record.model_id:
        raise ValueError(f"AI record {record.id!r} provenance.model_id must match model_id")
    if provenance.get("computed_feature_sha256") != canonical_json_sha256(
        raw_record.get("features")
    ):
        raise ValueError(
            f"AI record {record.id!r} computed_feature_sha256 does not match its feature payload"
        )


def _validate_public_ai_artifacts(
    raw_record: Mapping[str, Any], record: BenchmarkRecord, artifact_root: Path | None
) -> None:
    """Bind public AI provenance digests to archived prompt/response artifacts."""
    provenance = _mapping(raw_record.get("provenance"), f"AI record {record.id!r} provenance")
    artifacts = _mapping(
        provenance.get("artifacts"), f"AI record {record.id!r} provenance.artifacts"
    )
    expected = {
        "prompt": "prompt_sha256",
        "generation_config": "generation_config_sha256",
        "raw_response": "raw_response_sha256",
        "generated_html": "generated_html_sha256",
        "rendered_html": "rendered_html_sha256",
        "screenshot": "screenshot_sha256",
    }
    for artifact_name, digest_field in expected.items():
        observed = _validate_capture_artifact(
            artifacts.get(artifact_name),
            f"AI record {record.id!r} provenance.artifacts.{artifact_name}",
            artifact_root,
        )
        if observed != provenance.get(digest_field):
            raise ValueError(
                f"AI record {record.id!r} {artifact_name} artifact digest does not match {digest_field}"
            )
    feature_artifact = artifacts.get("computed_features")
    _validate_capture_artifact(
        feature_artifact,
        f"AI record {record.id!r} provenance.artifacts.computed_features",
        artifact_root,
    )
    if artifact_root is not None:
        feature_item = _mapping(
            feature_artifact, f"AI record {record.id!r} provenance.artifacts.computed_features"
        )
        feature_path = artifact_root / _relative_artifact_path(
            feature_item.get("path"),
            f"AI record {record.id!r} provenance.artifacts.computed_features.path",
        )
        try:
            with feature_path.open(encoding="utf-8") as handle:
                archived_features = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"AI record {record.id!r} computed_features artifact must be valid JSON"
            ) from exc
        if canonical_json_sha256(archived_features) != provenance.get("computed_feature_sha256"):
            raise ValueError(
                f"AI record {record.id!r} computed_features artifact does not match its feature payload"
            )


def _require_public_scoring_group_records(records: list[BenchmarkRecord]) -> None:
    """Require one declared scoring viewport capture per public group.

    Alternate mobile captures belong in the ledger. This makes the published
    score unit explicit and avoids silently averaging a group because a second
    diagnostic viewport happened to be included in the benchmark manifest.
    """
    groups: dict[str, int] = {}
    for record in records:
        groups[record.group_id] = groups.get(record.group_id, 0) + 1
    repeated = sorted(group_id for group_id, count in groups.items() if count != 1)
    if repeated:
        preview = ", ".join(repeated[:3])
        raise ValueError(
            "a public benchmark must contain exactly one scoring record per group; "
            f"retain alternate captures in the ledger (examples: {preview})"
        )


def _precheck_admitted_human_source_coverage(
    benchmark_manifest: Mapping[str, Any], sources: list[OpenReferenceSource]
) -> None:
    """Report a missing admitted source before downstream split diagnostics.

    The benchmark validator also rejects a test stratum that loses its human
    row. At the public-release boundary, the more actionable failure is that a
    curated source no longer has a scoring record. This lightweight check only
    recognizes well-formed human provenance bindings; the full schema and
    cross-artifact validation still run immediately afterwards.
    """
    raw_records = benchmark_manifest.get("records")
    if not isinstance(raw_records, list):
        return
    observed: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or raw_record.get("source") != "human":
            continue
        provenance = raw_record.get("provenance")
        if isinstance(provenance, Mapping):
            source_id = provenance.get("source_id")
            if isinstance(source_id, str) and source_id.strip():
                observed.add(source_id.strip())
    missing = sorted({source.id for source in sources}.difference(observed))
    if missing:
        raise ValueError(
            "every admitted public human source must have exactly one scoring record; "
            f"missing source records: {', '.join(missing[:3])}"
        )


def validate_public_release(
    source_manifest: Mapping[str, Any],
    capture_ledger: Mapping[str, Any],
    benchmark_manifest: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a complete public release without making empirical claims.

    ``artifact_root`` is optional because an archive verifier may have only the
    JSON indices. Supplying it additionally recomputes every saved HTML and
    screenshot digest.
    """
    sources = validate_open_reference_manifest(source_manifest)
    source_protocol, provenance_policy = _validate_public_source_metadata(source_manifest)
    raw_sources = source_manifest.get("sources")
    assert isinstance(raw_sources, list)  # Guaranteed by validate_open_reference_manifest.
    for raw_source in raw_sources:
        source_row = _mapping(raw_source, "source row")
        if provenance_policy["mode"] == "independent_reviews":
            _validate_provenance_reviews(source_row, int(provenance_policy["minimum"]))
        else:
            _validate_historical_provenance_evidence(source_row, int(provenance_policy["minimum"]))

    captures = _validate_capture_ledger(
        capture_ledger, source_manifest, sources, source_protocol, artifact_root
    )
    _precheck_admitted_human_source_coverage(benchmark_manifest, sources)
    records = validate_manifest(benchmark_manifest)
    extractor = _validate_public_benchmark_metadata(
        benchmark_manifest, source_manifest, capture_ledger
    )
    if extractor != capture_ledger.get("feature_extractor_version"):
        raise ValueError("benchmark feature_extractor_version does not match the capture ledger")
    source_metadata = _mapping(source_manifest.get("metadata"), "source metadata")
    benchmark_metadata = _mapping(benchmark_manifest.get("metadata"), "benchmark metadata")
    if benchmark_metadata.get("release_revision") != source_metadata.get("release_revision"):
        raise ValueError("source and benchmark release_revision values must match")
    scoring_viewport = str(benchmark_metadata["scoring_viewport"])
    _require_public_scoring_group_records(records)
    raw_records = benchmark_manifest.get("records")
    assert isinstance(raw_records, list)  # Guaranteed by validate_manifest.
    source_by_id = {source.id: source for source in sources}
    admitted_human_sources: set[str] = set()
    for raw_record, record in zip(raw_records, records, strict=True):
        raw_mapping = _mapping(raw_record, "benchmark record")
        if not raw_mapping.get("leakage_group_id"):
            raise ValueError(
                f"public benchmark record {record.id!r} requires an explicit leakage_group_id"
            )
        if record.viewport != scoring_viewport:
            raise ValueError(
                f"public benchmark record {record.id!r} must use the declared scoring_viewport"
            )
        if record.source == "human":
            _validate_public_human_record(
                raw_mapping, record, source_by_id, captures, scoring_viewport
            )
            admitted_human_sources.add(
                str(
                    _mapping(raw_mapping["provenance"], f"human record {record.id!r} provenance")[
                        "source_id"
                    ]
                )
            )
        else:
            _validate_public_ai_record(raw_mapping, record)
            provenance = _mapping(
                raw_mapping.get("provenance"), f"AI record {record.id!r} provenance"
            )
            if provenance.get("feature_extractor_version") != extractor:
                raise ValueError(
                    f"AI record {record.id!r} feature_extractor_version does not match the capture ledger"
                )
            _validate_public_ai_artifacts(raw_mapping, record, artifact_root)
    expected_human_sources = {source.id for source in sources}
    if admitted_human_sources != expected_human_sources:
        missing = sorted(expected_human_sources.difference(admitted_human_sources))
        extra = sorted(admitted_human_sources.difference(expected_human_sources))
        detail = []
        if missing:
            detail.append(f"missing source records: {', '.join(missing[:3])}")
        if extra:
            detail.append(f"unknown source records: {', '.join(extra[:3])}")
        raise ValueError(
            "every admitted public human source must have exactly one scoring record; "
            + "; ".join(detail)
        )

    return {
        "schema": PUBLIC_RELEASE_SCHEMA,
        "source_manifest_sha256": canonical_json_sha256(source_manifest),
        "capture_ledger_sha256": canonical_json_sha256(capture_ledger),
        "benchmark_manifest_sha256": canonical_json_sha256(benchmark_manifest),
        "release_revision": benchmark_metadata["release_revision"],
        "scoring_viewport": scoring_viewport,
        "feature_extractor_version": extractor,
        "provenance_policy_mode": provenance_policy["mode"],
        "artifact_hashes_checked": artifact_root is not None,
        "n_sources": len(sources),
        "n_captures": len(captures),
        "n_benchmark_records": len(records),
        "n_human_records": sum(record.source == "human" for record in records),
        "n_ai_records": sum(record.source == "ai" for record in records),
        "note": (
            "Release completeness and cross-artifact provenance passed. This audit does not prove "
            "human authorship, preference, accessibility, or benchmark effectiveness."
        ),
    }
