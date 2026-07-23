"""Protocol-level audits for the frozen WebHumanBench v1 release candidate."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .release import sha256_file

PROTOCOL_AUDIT_SCHEMA = "webmark_webhumanbench_v1_protocol_audit_v1"
TEMPORAL_CUTOFF = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _parse_timestamp(value: object) -> datetime:
    text = str(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {text!r}")
    return parsed.astimezone(timezone.utc)


def _load_messages(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError(f"prompt artifact must contain a non-empty message list: {path}")
    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError(f"prompt message must be an object: {path}")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if not role or not content:
            raise ValueError(f"prompt message requires role and content: {path}")
        messages.append({"role": role, "content": content})
    return messages


def _counter_rows(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "count": counter[key]} for key in sorted(counter)]


def build_protocol_audit(
    source_manifest: Mapping[str, Any],
    capture_ledger: Mapping[str, Any],
    benchmark_manifest: Mapping[str, Any],
    source_run_ledger: Mapping[str, Any],
    *,
    capture_root: Path,
) -> dict[str, Any]:
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source manifest requires non-empty sources")
    source_ids = {str(source["id"]) for source in sources}
    if len(source_ids) != len(sources):
        raise ValueError("source manifest contains duplicate source ids")

    commits = [_parse_timestamp(source["commit_authored_at"]) for source in sources]
    if any(commit >= TEMPORAL_CUTOFF for commit in commits):
        raise ValueError("historical-reference source has a commit at or after the cutoff")
    commit_years = Counter(str(commit.year) for commit in commits)

    vendor_container = capture_ledger.get("vendor_receipts")
    if not isinstance(vendor_container, Mapping):
        raise ValueError("capture ledger requires vendor_receipts")
    vendor_rows = vendor_container.get("records")
    if not isinstance(vendor_rows, list) or not vendor_rows:
        raise ValueError("capture ledger requires vendor receipt records")
    vendor_ids = {str(row["source_id"]) for row in vendor_rows}
    if vendor_ids != source_ids or len(vendor_ids) != len(vendor_rows):
        raise ValueError("vendor receipt records do not map one-to-one to reference sources")

    vendor_assets = [asset for row in vendor_rows for asset in row.get("vendor_assets", [])]
    removed_scripts = [
        script for row in vendor_rows for script in row.get("removed_external_scripts", [])
    ]
    content_types = Counter(str(asset.get("content_type", "unknown")) for asset in vendor_assets)
    sources_with_assets = [
        str(row["source_id"]) for row in vendor_rows if row.get("vendor_assets")
    ]
    modified_entrypoints = [
        str(row["source_id"])
        for row in vendor_rows
        if str(row.get("original_entrypoint_sha256"))
        != str(row.get("snapshot_entrypoint_sha256"))
    ]

    planned_rows = source_run_ledger.get("records")
    if source_run_ledger.get("schema") != "webhumanbench_v1_ai_source_run_ledger_v1":
        raise ValueError("unexpected AI source-run ledger schema")
    if not isinstance(planned_rows, list) or not planned_rows:
        raise ValueError("AI source-run ledger requires records")
    planned = {str(row["id"]): row for row in planned_rows}
    if len(planned) != len(planned_rows):
        raise ValueError("AI source-run ledger contains duplicate ids")

    ai_rows = [
        row for row in benchmark_manifest.get("records", []) if row.get("source") == "ai"
    ]
    if not ai_rows:
        raise ValueError("benchmark manifest has no generated challenge rows")
    required_clauses = {
        "production_like_system_condition": "production-like HTML/CSS webpages",
        "self_contained_no_external_assets": "No external assets, scripts, icon libraries, web fonts, iframes, or network calls.",
        "literal_css_extractor_condition": "Use literal CSS values",
        "human_authored_style_condition": "plausible for a modern human-authored website",
    }
    clause_counts: Counter[str] = Counter()
    system_prompts: set[str] = set()
    user_template_hashes: set[str] = set()
    generated_times: list[datetime] = []
    retained_models: Counter[str] = Counter()
    retained_page_types: Counter[str] = Counter()

    for row in ai_rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("generated challenge row requires provenance")
        prompt_id = str(provenance.get("prompt_id", ""))
        if prompt_id not in planned:
            raise ValueError(f"retained prompt id is absent from source-run ledger: {prompt_id}")
        plan = planned[prompt_id]
        if str(plan["model_id"]) != str(provenance.get("model_id")):
            raise ValueError(f"model id mismatch for prompt {prompt_id}")
        if str(plan["page_type"]) != str(row.get("page_type")):
            raise ValueError(f"page-type mismatch for prompt {prompt_id}")

        artifacts = provenance.get("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("prompt"), Mapping):
            raise ValueError(f"generated challenge row lacks a prompt artifact: {prompt_id}")
        prompt_artifact = artifacts["prompt"]
        prompt_path = capture_root / str(prompt_artifact["path"])
        if sha256_file(prompt_path) != str(prompt_artifact["sha256"]):
            raise ValueError(f"prompt hash mismatch: {prompt_path}")
        messages = _load_messages(prompt_path)
        system = "\n".join(message["content"] for message in messages if message["role"] == "system")
        user = "\n".join(message["content"] for message in messages if message["role"] == "user")
        if not system or not user:
            raise ValueError(f"prompt requires system and user messages: {prompt_id}")
        system_prompts.add(system)
        combined = f"{system}\n{user}"
        for name, clause in required_clauses.items():
            if clause in combined:
                clause_counts[name] += 1

        expected_lines = (
            f"Scenario/theme: {plan['scenario']}",
            f"Page type label: {plan['page_type']}",
            f"Variation seed: {plan['seed']}",
        )
        if any(line not in user for line in expected_lines):
            raise ValueError(f"prompt does not match source-run metadata: {prompt_id}")
        normalized = re.sub(r"Scenario/theme: .*", "Scenario/theme: <SCENARIO>", user)
        normalized = re.sub(r"Variation seed: \d+", "Variation seed: <SEED>", normalized)
        user_template_hashes.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        generated_times.append(_parse_timestamp(provenance["generated_at"]))
        retained_models[str(provenance["model_id"])] += 1
        retained_page_types[str(row["page_type"])] += 1

    planned_models = Counter(str(row["model_id"]) for row in planned_rows)
    planned_page_types = Counter(str(row["page_type"]) for row in planned_rows)
    scenarios_by_type: dict[str, set[str]] = {}
    for row in planned_rows:
        scenarios_by_type.setdefault(str(row["page_type"]), set()).add(str(row["scenario"]))

    n_ai = len(ai_rows)
    return {
        "schema": PROTOCOL_AUDIT_SCHEMA,
        "reference_temporal_provenance": {
            "n_sources": len(sources),
            "cutoff_utc": TEMPORAL_CUTOFF.isoformat(),
            "earliest_commit_authored_at": min(commits).isoformat(),
            "latest_commit_authored_at": max(commits).isoformat(),
            "commit_year_counts": _counter_rows(commit_years, "year"),
            "sources_committed_in_2022": commit_years["2022"],
            "all_commits_before_cutoff": True,
            "boundary": (
                "The cutoff applies to the pinned repository commit. It does not date every "
                "externally hosted visual asset later frozen for local capture."
            ),
        },
        "reference_capture_interventions": {
            "captured_at": str(capture_ledger.get("captured_at")),
            "n_sources": len(vendor_rows),
            "sources_with_vendored_assets": len(sources_with_assets),
            "vendored_asset_count": len(vendor_assets),
            "vendored_asset_content_types": _counter_rows(content_types, "content_type"),
            "sources_with_modified_snapshot_entrypoints": len(modified_entrypoints),
            "sources_with_removed_external_scripts": sum(
                bool(row.get("removed_external_scripts")) for row in vendor_rows
            ),
            "removed_external_script_count": len(removed_scripts),
            "source_ids_with_vendored_assets": sorted(sources_with_assets),
            "source_ids_with_modified_snapshot_entrypoints": sorted(modified_entrypoints),
            "boundary": (
                "Vendoring makes the 2026 capture reproducible but does not establish that the "
                "fetched bytes equal the bytes served at the pinned commit date."
            ),
        },
        "generated_prompt_protocol": {
            "planned_pages": len(planned_rows),
            "audited_retained_prompt_artifacts": n_ai,
            "unique_system_prompts": len(system_prompts),
            "normalized_user_templates": len(user_template_hashes),
            "required_clause_coverage": {
                name: {"count": clause_counts[name], "fraction": clause_counts[name] / n_ai}
                for name in required_clauses
            },
            "generated_at_start": min(generated_times).isoformat(),
            "generated_at_end": max(generated_times).isoformat(),
            "planned_by_model": _counter_rows(planned_models, "model_id"),
            "retained_by_model": _counter_rows(retained_models, "model_id"),
            "planned_by_page_type": _counter_rows(planned_page_types, "page_type"),
            "retained_by_page_type": _counter_rows(retained_page_types, "page_type"),
            "unique_scenarios_by_page_type": [
                {"page_type": page_type, "count": len(scenarios_by_type[page_type])}
                for page_type in sorted(scenarios_by_type)
            ],
            "boundary": (
                "The challenge cohort is prompt- and pipeline-conditioned. Every retained prompt "
                "requests production-like, self-contained, literal-CSS output and explicitly asks "
                "for a plausible modern human-authored style; the cohort is not an uncontrolled "
                "sample of model generations."
            ),
        },
        "interpretation": (
            "The release candidate is hash-auditable, but auditability does not remove temporal "
            "asset, prompt, content, provider, or capture-pipeline confounding."
        ),
    }

