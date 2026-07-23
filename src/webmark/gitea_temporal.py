"""Fail-closed provenance helpers for versioned Gitea webpage sources."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .open_reference import PERMISSIVE_LICENSES

QUERY_SCHEMA = "webmark_gitea_temporal_queries_v1"
CANDIDATE_SCHEMA = "webmark_gitea_temporal_candidates_v1"
AUDIT_SCHEMA = "webmark_gitea_temporal_source_audit_v1"

LICENSE_NAMES = frozenset({"license", "license.md", "license.txt", "copying"})
STATIC_SKIP_PARTS = frozenset(
    {
        ".git",
        "coverage",
        "example",
        "examples",
        "fixture",
        "fixtures",
        "node_modules",
        "test",
        "tests",
        "vendor",
    }
)
STATIC_ROOTS = frozenset(
    {
        "",
        "_site",
        "app",
        "build",
        "dist",
        "docs",
        "frontend",
        "out",
        "public",
        "site",
        "static",
        "web",
        "website",
        "www",
    }
)


def parse_timestamp(value: Any, field: str) -> datetime:
    """Parse and normalize an ISO-8601 timestamp."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} requires a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} requires an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} requires a timezone")
    return parsed.astimezone(UTC)


def utc_text(value: datetime) -> str:
    """Serialize a timezone-aware timestamp in canonical UTC form."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def repository_is_eligible(repository: Mapping[str, Any], *, cutoff: datetime, cohort: str) -> bool:
    """Apply cohort and repository-integrity filters before expensive auditing."""
    if cohort not in {"pre", "post"}:
        raise ValueError("cohort must be 'pre' or 'post'")
    if any(bool(repository.get(field)) for field in ("archived", "empty", "fork", "mirror", "private", "template")):
        return False
    full_name = repository.get("full_name")
    default_branch = repository.get("default_branch")
    if not isinstance(full_name, str) or full_name.count("/") != 1 or not default_branch:
        return False
    try:
        created_at = parse_timestamp(repository.get("created_at"), "repository.created_at")
    except ValueError:
        return False
    return created_at < cutoff if cohort == "pre" else created_at >= cutoff


def candidate_from_repository(
    repository: Mapping[str, Any],
    *,
    host_id: str,
    page_type: str,
    query: str,
    cutoff: datetime,
    cohort: str,
) -> dict[str, Any] | None:
    """Normalize one repository search hit without promoting its page-type label."""
    if not repository_is_eligible(repository, cutoff=cutoff, cohort=cohort):
        return None
    full_name = str(repository["full_name"])
    repository_id = repository.get("id")
    if not isinstance(repository_id, int) or isinstance(repository_id, bool):
        return None
    return {
        "candidate_id": f"{host_id}:{repository_id}:{page_type}",
        "repository_identity": f"{host_id}:{repository_id}",
        "host_id": host_id,
        "repository_id": repository_id,
        "repository": full_name,
        "repository_url": str(repository.get("html_url") or ""),
        "repository_created_at": utc_text(parse_timestamp(repository["created_at"], "repository.created_at")),
        "repository_updated_at": utc_text(parse_timestamp(repository["updated_at"], "repository.updated_at")),
        "default_branch_at_discovery": str(repository["default_branch"]),
        "page_type_candidate": page_type,
        "discovery_queries": [query],
        "language_candidate": repository.get("language"),
        "description": repository.get("description") or "",
        "stars": int(repository.get("stars_count") or 0),
        "size_kib": int(repository.get("size") or 0),
        "cohort_candidate": cohort,
        "page_type_verified": False,
        "entrypoint_verified": False,
        "render_verified": False,
    }


def merge_candidate_hit(existing: dict[str, Any], query: str) -> None:
    """Record repeated discovery without duplicating a candidate group."""
    queries = existing.setdefault("discovery_queries", [])
    if query not in queries:
        queries.append(query)
        queries.sort()


def commit_evidence(payload: Any, *, cutoff: datetime, cohort: str) -> dict[str, str]:
    """Validate a pinned branch-head commit against a temporal cohort."""
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], Mapping):
        raise ValueError("commit endpoint returned no commit object")
    row = payload[0]
    sha = row.get("sha")
    if not isinstance(sha, str) or len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("commit endpoint did not return a full lowercase SHA")
    commit = row.get("commit")
    author = commit.get("author") if isinstance(commit, Mapping) else None
    committer = commit.get("committer") if isinstance(commit, Mapping) else None
    authored = parse_timestamp(author.get("date") if isinstance(author, Mapping) else None, "commit.author.date")
    committed = parse_timestamp(
        committer.get("date") if isinstance(committer, Mapping) else None,
        "commit.committer.date",
    )
    if cohort == "pre":
        valid = authored < cutoff and committed < cutoff
    elif cohort == "post":
        valid = authored >= cutoff and committed >= cutoff
    else:
        raise ValueError("cohort must be 'pre' or 'post'")
    if not valid:
        raise ValueError(f"commit timestamps do not satisfy the {cohort!r} cohort cutoff")
    return {
        "commit_sha": sha,
        "commit_authored_at": utc_text(authored),
        "commit_committed_at": utc_text(committed),
    }


def tree_paths(payloads: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    """Validate paginated recursive trees and return one stable path set."""
    if not payloads:
        raise ValueError("tree endpoint returned no pages")
    tree_sha: str | None = None
    paths: list[str] = []
    for page_number, payload in enumerate(payloads, start=1):
        observed_sha = payload.get("sha")
        if not isinstance(observed_sha, str) or len(observed_sha) != 40:
            raise ValueError(f"tree page {page_number} has no full tree SHA")
        if tree_sha is None:
            tree_sha = observed_sha
        elif observed_sha != tree_sha:
            raise ValueError("paginated tree responses disagree on tree SHA")
        rows = payload.get("tree")
        if not isinstance(rows, list):
            raise ValueError(f"tree page {page_number} has no tree list")
        for row in rows:
            if isinstance(row, Mapping) and row.get("type") == "blob" and isinstance(row.get("path"), str):
                paths.append(str(row["path"]))
    if len(paths) != len(set(paths)):
        raise ValueError("paginated tree contains duplicate blob paths")
    return str(tree_sha), sorted(paths)


def license_paths(paths: Sequence[str]) -> list[str]:
    """Return root-level license files in deterministic preference order."""
    return sorted(
        (path for path in paths if "/" not in path and path.lower() in LICENSE_NAMES),
        key=lambda path: (path.lower() != "license", len(path), path.lower()),
    )


def detect_license_spdx(text: str) -> str | None:
    """Recognize the small permissive-license allowlist from pinned license text."""
    normalized = " ".join(text.lower().split())
    if "apache license" in normalized and "version 2.0" in normalized:
        return "Apache-2.0"
    if "cc0 1.0 universal" in normalized or (
        "creative commons" in normalized and "public domain dedication" in normalized
    ):
        return "CC0-1.0"
    if "permission is hereby granted, free of charge" in normalized and "the software is provided \"as is\"" in normalized:
        return "MIT"
    if "redistribution and use in source and binary forms" in normalized:
        if "neither the name" in normalized:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "permission to use, copy, modify, and/or distribute this software" in normalized and "the software is provided \"as is\"" in normalized:
        return "ISC"
    return None


def static_entrypoints(paths: Sequence[str]) -> list[str]:
    """Select root-like committed HTML entrypoints while excluding fixtures."""
    candidates: list[str] = []
    for path in paths:
        parts = path.split("/")
        lowered = [part.lower() for part in parts]
        if any(part in STATIC_SKIP_PARTS for part in lowered[:-1]):
            continue
        if lowered[-1] not in {"index.html", "index.htm"}:
            continue
        parents = lowered[:-1]
        parent = "/".join(parents)
        root_like = (
            parent in STATIC_ROOTS
            or (
                1 <= len(parents) <= 3
                and (parents[0] in STATIC_ROOTS or parents[-1] in STATIC_ROOTS)
            )
        )
        if not root_like:
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda path: (path.count("/"), len(path), path.lower()))


def package_paths(paths: Sequence[str]) -> list[str]:
    """Return root-like package manifests as optional build evidence."""
    allowed = {"package.json", "docs/package.json", "site/package.json", "website/package.json"}
    return sorted((path for path in paths if path.lower() in allowed), key=lambda path: (path.count("/"), path))


class GiteaClient:
    """Minimal retrying client for public Gitea evidence endpoints."""

    def __init__(
        self,
        *,
        api_base: str,
        web_base: str,
        token: str | None = None,
        timeout_s: int = 30,
        retries: int = 3,
    ) -> None:
        self.api_base = self._https_base(api_base, "api_base")
        self.web_base = self._https_base(web_base, "web_base")
        self.token = token
        self.timeout_s = timeout_s
        self.retries = retries

    @staticmethod
    def _https_base(value: str, field: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError(f"{field} must be an HTTPS origin or path")
        return value.rstrip("/")

    def _read(self, url: str, *, accept: str) -> bytes:
        headers = {"Accept": accept, "User-Agent": "webhumanbench-temporal-source-audit/1"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        request = Request(url, headers=headers)
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:  # nosec B310 - configured HTTPS evidence host
                    return response.read()
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 == self.retries:
                    raise
            except (TimeoutError, URLError, OSError):
                if attempt + 1 == self.retries:
                    raise
            time.sleep(1 + attempt)
        raise RuntimeError("unreachable retry state")

    def json(self, path: str, query: Mapping[str, Any] | None = None) -> Any:
        raw = self.bytes(path, query=query, accept="application/json")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gitea endpoint returned invalid JSON") from exc

    def bytes(
        self,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        accept: str = "application/octet-stream",
    ) -> bytes:
        """Read an API path as bytes with the same fixed retry policy."""
        suffix = f"?{urlencode(query)}" if query else ""
        return self._read(f"{self.api_base}/{path.lstrip('/')}{suffix}", accept=accept)

    def raw(self, repository: str, commit_sha: str, path: str) -> bytes:
        owner, name = repository.split("/", 1)
        encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
        url = (
            f"{self.web_base}/{quote(owner, safe='')}/{quote(name, safe='')}/raw/commit/"
            f"{quote(commit_sha, safe='')}/{encoded_path}"
        )
        return self._read(url, accept="text/plain, application/octet-stream")

    def repository_path(self, repository: str, suffix: str = "") -> str:
        owner, name = repository.split("/", 1)
        base = f"repos/{quote(owner, safe='')}/{quote(name, safe='')}"
        return f"{base}/{suffix.lstrip('/')}" if suffix else base

    def web_url(self, repository: str, suffix: str = "") -> str:
        owner, name = repository.split("/", 1)
        base = f"{self.web_base}/{quote(owner, safe='')}/{quote(name, safe='')}"
        return f"{base}/{suffix.lstrip('/')}" if suffix else base


def allowed_license(spdx: str | None) -> bool:
    """Keep license policy centralized with the open-reference validator."""
    return spdx in PERMISSIVE_LICENSES
