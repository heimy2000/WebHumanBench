"""Validation for an open, mobile-first human-reference source manifest."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlparse

from .human_likeness import MOBILE_MAX_WIDTH, viewport_width
from .page_type import PAGE_TYPES
from .pinned_build import BuildRecipe, build_recipe_from_dict

SCHEMA = "webmark_open_mobile_reference_v1"
HISTORICAL_EVIDENCE_SCHEMA = "webmark_open_mobile_reference_v2"
CONTEMPORARY_EVIDENCE_SCHEMA = "webmark_open_mobile_reference_v3"
SUPPORTED_SCHEMAS = frozenset(
    {SCHEMA, HISTORICAL_EVIDENCE_SCHEMA, CONTEMPORARY_EVIDENCE_SCHEMA}
)
PERMISSIVE_LICENSES = frozenset({"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "ISC", "MIT"})
PRIMARY_MOBILE_VIEWPORTS = ("390x844", "430x932")
TEMPORAL_POLICIES = frozenset({"before_cutoff", "on_or_after_cutoff"})
CAPTURE_METHODS = frozenset({
    "remote_observation",
    "pinned_local_static_checkout",
    "pinned_local_static_snapshot",
    "pinned_local_build",
})
PINNED_LOCAL_CAPTURE_METHODS = frozenset({
    "pinned_local_static_checkout",
    "pinned_local_static_snapshot",
    "pinned_local_build",
})
SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class OpenReferenceSource:
    """Pinned open-source webpage source and its reproducible mobile capture plan."""

    id: str
    page_type: str
    group_id: str
    repository_url: str
    repository_created_at: str
    commit_sha: str
    commit_authored_at: str
    license_spdx: str
    license_url: str
    entrypoint: str
    entrypoint_evidence_url: str
    build_command: str
    capture_url: str
    capture_commit_evidence_url: str
    capture_method: str
    build_recipe: BuildRecipe | None
    curation_reviewer_id: str | None
    curation_reviewed_at: str | None
    human_provenance_urls: tuple[str, ...]
    viewports: tuple[str, ...]


def _https_url(value: Any, field: str, source_id: str) -> str:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"source {source_id!r} requires an HTTPS {field}")
    return text


def _source_identifier(value: Any, field: str) -> str:
    text = str(value)
    if not SOURCE_ID_RE.fullmatch(text):
        raise ValueError(f"{field} must use a portable identifier of up to 128 letters, digits, dots, underscores, or hyphens")
    return text


def _repository_url(value: Any, source_id: str) -> str:
    url = _https_url(value, "repository_url", source_id)
    parsed = urlparse(url)
    if not parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError(f"source {source_id!r} repository_url must identify one repository without a query or fragment")
    return url


def _canonical_repository_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path.lower()}"


def _parsed_timestamp(value: Any, field: str, source_id: str) -> datetime:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"source {source_id!r} requires an ISO-8601 {field}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"source {source_id!r} requires a timezone-aware {field}")
    return parsed.astimezone(UTC)


def _timestamp(value: Any, field: str, source_id: str) -> str:
    text = str(value)
    _parsed_timestamp(text, field, source_id)
    return text


def _relative_source_path(value: Any, field: str, source_id: str) -> str:
    """Require a portable repository-relative source path."""
    text = str(value)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError(f"source {source_id!r} requires a repository-relative {field}")
    return text


def _pinned_evidence_url(value: Any, field: str, source_id: str, commit_sha: str) -> str:
    """Require an HTTPS evidence link that visibly names the pinned revision."""
    url = _https_url(value, field, source_id)
    parsed = urlparse(url)
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    query_values = [value for _key, value in parse_qsl(parsed.query)]
    if commit_sha not in path_segments and commit_sha not in query_values:
        raise ValueError(f"source {source_id!r} {field} must identify commit_sha")
    return url


def _provenance_urls(value: Any, source_id: str) -> tuple[str, ...]:
    """Normalize non-empty curator evidence URLs without treating them as proof."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"source {source_id!r} requires non-empty human_provenance_urls")
    return tuple(_https_url(url, "human_provenance_urls", source_id) for url in value)


def _observable_provenance_evidence(
    value: Any, source_id: str, commit_sha: str, *, minimum_kinds: int
) -> None:
    """Validate non-visual evidence for an open-source temporal proxy label.

    This mode intentionally records observable source facts instead of
    inventing independent human review identities. It establishes neither true
    authorship nor an absence of AI assistance.
    """
    if not isinstance(value, list) or len(value) < minimum_kinds:
        raise ValueError(
            f"source {source_id!r} requires at least {minimum_kinds} historical provenance evidence entries"
        )
    kinds: set[str] = set()
    for index, raw_evidence in enumerate(value):
        if not isinstance(raw_evidence, Mapping):
            raise ValueError(f"source {source_id!r} provenance_evidence[{index}] must be an object")
        kind = str(raw_evidence.get("kind", "")).strip()
        if kind not in {"pinned_git_history", "source_project_identity"}:
            raise ValueError(
                f"source {source_id!r} provenance_evidence[{index}].kind must be "
                "'pinned_git_history' or 'source_project_identity'"
            )
        kinds.add(kind)
        urls = raw_evidence.get("evidence_urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError(f"source {source_id!r} provenance_evidence[{index}] requires evidence_urls")
        normalized = [
            _https_url(url, f"provenance_evidence[{index}].evidence_urls", source_id)
            for url in urls
        ]
        if kind == "pinned_git_history" and not any(
            commit_sha in urlparse(url).path.split("/") or commit_sha in urlparse(url).query
            for url in normalized
        ):
            raise ValueError(
                f"source {source_id!r} pinned_git_history evidence must visibly identify commit_sha"
            )
    if len(kinds) < minimum_kinds:
        raise ValueError(
            f"source {source_id!r} observable provenance evidence requires {minimum_kinds} distinct kinds"
        )


def source_from_dict(
    raw: Mapping[str, Any], *, historical_evidence: bool = False, minimum_evidence_kinds: int = 2
) -> OpenReferenceSource:
    """Parse a source row with repository, licensing, and mobile capture evidence."""
    required = (
        "id", "page_type", "group_id", "repository_url", "repository_created_at", "commit_sha", "commit_authored_at", "license_spdx",
        "license_url", "entrypoint", "entrypoint_evidence_url", "build_command", "capture_url", "capture_commit_evidence_url",
        "viewports",
    )
    if not historical_evidence:
        required += ("curation_reviewer_id", "curation_reviewed_at", "human_provenance_urls")
    missing = [field for field in required if not raw.get(field)]
    if missing:
        raise ValueError(f"open-reference source is missing: {', '.join(missing)}")
    source_id = _source_identifier(raw["id"], "source id")
    page_type = str(raw["page_type"])
    if page_type not in PAGE_TYPES:
        raise ValueError(f"source {source_id!r} has unsupported page_type {page_type!r}")
    commit_sha = str(raw["commit_sha"])
    if not re.fullmatch(r"[0-9a-f]{7,64}", commit_sha):
        raise ValueError(f"source {source_id!r} requires a pinned hexadecimal commit_sha")
    license_spdx = str(raw["license_spdx"])
    if license_spdx not in PERMISSIVE_LICENSES:
        raise ValueError(
            f"source {source_id!r} license {license_spdx!r} is not in the allowed open-source set"
        )
    capture_method = str(raw.get("capture_method", "remote_observation"))
    if capture_method not in CAPTURE_METHODS:
        raise ValueError(
            f"source {source_id!r} capture_method must be one of {sorted(CAPTURE_METHODS)!r}"
        )
    build_recipe: BuildRecipe | None = None
    if capture_method == "pinned_local_build":
        build_recipe = build_recipe_from_dict(raw.get("build_recipe"), source_id)
        declared_command = str(raw["build_command"]).strip()
        expected_command = build_recipe.build_command
        if declared_command != expected_command:
            raise ValueError(
                f"source {source_id!r} build_command must be {expected_command!r} for its pinned build recipe"
            )
    elif raw.get("build_recipe") is not None:
        raise ValueError(
            f"source {source_id!r} build_recipe is valid only for capture_method 'pinned_local_build'"
        )
    viewports = raw["viewports"]
    if not isinstance(viewports, list) or not viewports:
        raise ValueError(f"source {source_id!r} requires a non-empty viewports list")
    normalized_viewports = tuple(str(viewport) for viewport in viewports)
    if len(normalized_viewports) != len(set(normalized_viewports)):
        raise ValueError(f"source {source_id!r} viewports must not contain duplicates")
    missing_primary = [viewport for viewport in PRIMARY_MOBILE_VIEWPORTS if viewport not in normalized_viewports]
    if missing_primary:
        raise ValueError(f"source {source_id!r} requires primary mobile viewports: {', '.join(missing_primary)}")
    if historical_evidence:
        _observable_provenance_evidence(
            raw.get("provenance_evidence"), source_id, commit_sha, minimum_kinds=minimum_evidence_kinds
        )
    return OpenReferenceSource(
        id=source_id,
        page_type=page_type,
        group_id=_source_identifier(raw["group_id"], f"source {source_id!r} group_id"),
        repository_url=_repository_url(raw["repository_url"], source_id),
        repository_created_at=_timestamp(raw["repository_created_at"], "repository_created_at", source_id),
        commit_sha=commit_sha,
        commit_authored_at=_timestamp(raw["commit_authored_at"], "commit_authored_at", source_id),
        license_spdx=license_spdx,
        license_url=_pinned_evidence_url(raw["license_url"], "license_url", source_id, commit_sha),
        entrypoint=_relative_source_path(raw["entrypoint"], "entrypoint", source_id),
        entrypoint_evidence_url=_pinned_evidence_url(
            raw["entrypoint_evidence_url"], "entrypoint_evidence_url", source_id, commit_sha
        ),
        build_command=str(raw["build_command"]),
        capture_url=_https_url(raw["capture_url"], "capture_url", source_id),
        capture_commit_evidence_url=_pinned_evidence_url(
            raw["capture_commit_evidence_url"], "capture_commit_evidence_url", source_id, commit_sha
        ),
        capture_method=capture_method,
        build_recipe=build_recipe,
        curation_reviewer_id=(
            str(raw["curation_reviewer_id"]) if not historical_evidence else None
        ),
        curation_reviewed_at=(
            _timestamp(raw["curation_reviewed_at"], "curation_reviewed_at", source_id)
            if not historical_evidence
            else None
        ),
        human_provenance_urls=(
            _provenance_urls(raw["human_provenance_urls"], source_id) if not historical_evidence else ()
        ),
        viewports=normalized_viewports,
    )


def validate_open_reference_manifest(manifest: Mapping[str, Any]) -> list[OpenReferenceSource]:
    """Validate open licensing, reproducible source pinning, and mobile coverage."""
    schema = manifest.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"manifest schema must be one of {sorted(SUPPORTED_SCHEMAS)!r}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("manifest requires metadata")
    for field in (
        "dataset_name",
        "version",
        "manifest_license",
        "capture_browser",
        "temporal_cutoff",
        "primary_mobile_viewports",
        "required_page_types",
        "min_sources_per_page_type",
    ):
        if not metadata.get(field):
            raise ValueError(f"manifest metadata requires {field!r}")
    required_page_types = metadata["required_page_types"]
    if not isinstance(required_page_types, list) or not required_page_types:
        raise ValueError("metadata required_page_types must be a non-empty list")
    unknown_page_types = set(required_page_types).difference(PAGE_TYPES)
    if unknown_page_types:
        raise ValueError(f"unsupported required_page_types: {sorted(unknown_page_types)}")
    min_sources = metadata["min_sources_per_page_type"]
    if not isinstance(min_sources, int) or min_sources <= 0:
        raise ValueError("metadata min_sources_per_page_type must be a positive integer")
    cutoff = _parsed_timestamp(metadata["temporal_cutoff"], "temporal_cutoff", "metadata")
    temporal_policy = str(metadata.get("temporal_policy", "before_cutoff"))
    if temporal_policy not in TEMPORAL_POLICIES:
        raise ValueError(f"metadata temporal_policy must be one of {sorted(TEMPORAL_POLICIES)!r}")
    primary_viewports = metadata["primary_mobile_viewports"]
    if not isinstance(primary_viewports, list) or tuple(primary_viewports) != PRIMARY_MOBILE_VIEWPORTS:
        raise ValueError(f"metadata primary_mobile_viewports must be {list(PRIMARY_MOBILE_VIEWPORTS)!r}")
    historical_evidence = schema in {
        HISTORICAL_EVIDENCE_SCHEMA,
        CONTEMPORARY_EVIDENCE_SCHEMA,
    }
    minimum_evidence_kinds = 2
    if historical_evidence:
        policy = metadata.get("provenance_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("observable-evidence metadata requires provenance_policy")
        expected_mode = (
            "historical_open_source_evidence_v1"
            if schema == HISTORICAL_EVIDENCE_SCHEMA
            else "contemporary_open_source_evidence_v1"
        )
        if policy.get("mode") != expected_mode:
            raise ValueError(
                "observable-evidence metadata provenance_policy.mode must be "
                f"{expected_mode!r}"
            )
        minimum_evidence_kinds = policy.get("min_distinct_evidence_kinds")
        if not isinstance(minimum_evidence_kinds, int) or minimum_evidence_kinds < 2:
            raise ValueError(
                "observable-evidence provenance_policy.min_distinct_evidence_kinds must be at least 2"
            )
        if schema == HISTORICAL_EVIDENCE_SCHEMA and temporal_policy != "before_cutoff":
            raise ValueError("historical-evidence manifests require temporal_policy 'before_cutoff'")
        if schema == CONTEMPORARY_EVIDENCE_SCHEMA and temporal_policy != "on_or_after_cutoff":
            raise ValueError(
                "contemporary-evidence manifests require temporal_policy 'on_or_after_cutoff'"
            )
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("manifest requires a non-empty sources list")
    sources = [
        source_from_dict(
            raw, historical_evidence=historical_evidence, minimum_evidence_kinds=minimum_evidence_kinds
        )
        for raw in raw_sources
    ]
    if metadata.get("release_status") == "public":
        remote_sources = [source.id for source in sources if source.capture_method not in PINNED_LOCAL_CAPTURE_METHODS]
        if remote_sources:
            raise ValueError(
                "a public manifest requires a pinned local capture_method for every source; "
                f"remote-only examples: {', '.join(remote_sources[:3])}"
            )
    for source in sources:
        repository_created_at = _parsed_timestamp(source.repository_created_at, "repository_created_at", source.id)
        commit_authored_at = _parsed_timestamp(source.commit_authored_at, "commit_authored_at", source.id)
        if temporal_policy == "before_cutoff":
            if repository_created_at >= cutoff:
                raise ValueError(f"source {source.id!r} repository_created_at must precede temporal_cutoff")
            if commit_authored_at >= cutoff:
                raise ValueError(f"source {source.id!r} commit_authored_at must precede temporal_cutoff")
        else:
            if repository_created_at < cutoff:
                raise ValueError(f"source {source.id!r} repository_created_at must be on or after temporal_cutoff")
            if commit_authored_at < cutoff:
                raise ValueError(f"source {source.id!r} commit_authored_at must be on or after temporal_cutoff")
    ids = [source.id for source in sources]
    groups = [source.group_id for source in sources]
    repositories = [_canonical_repository_url(source.repository_url) for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source ids must be unique")
    if len(groups) != len(set(groups)):
        raise ValueError("reference group_id values must be unique to avoid duplicate sites")
    if len(repositories) != len(set(repositories)):
        raise ValueError("repository_url values must be unique to avoid duplicate source repositories")
    for page_type in required_page_types:
        count = sum(source.page_type == page_type for source in sources)
        if count < min_sources:
            raise ValueError(
                f"page type {page_type!r} has {count} sources; minimum is {min_sources}"
            )
    return sources


def open_reference_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return concise, serializable coverage metadata after validation."""
    sources = validate_open_reference_manifest(manifest)
    mobile_captures = sum(
        1 for source in sources for viewport in source.viewports if viewport_width(viewport) <= MOBILE_MAX_WIDTH
    )
    total_captures = sum(len(source.viewports) for source in sources)
    return {
        "schema": manifest["schema"],
        "dataset_name": manifest["metadata"]["dataset_name"],
        "version": manifest["metadata"]["version"],
        "temporal_cutoff": manifest["metadata"]["temporal_cutoff"],
        "temporal_policy": manifest["metadata"].get("temporal_policy", "before_cutoff"),
        "primary_mobile_viewports": list(PRIMARY_MOBILE_VIEWPORTS),
        "n_sources": len(sources),
        "n_mobile_captures": mobile_captures,
        "mobile_capture_share": mobile_captures / total_captures,
        "page_types": sorted({source.page_type for source in sources}),
        "licenses": sorted({source.license_spdx for source in sources}),
    }
