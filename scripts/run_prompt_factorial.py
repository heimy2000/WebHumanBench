#!/usr/bin/env python3
"""Run the matched 2x2 WebHumanBench prompt-sensitivity experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capture_open_mobile_reference import (
    DEFAULT_CHROME,
    FEATURE_EXTRACTOR_VERSION,
    _computed_feature_script,
    _local_only_route,
    _StaticCheckoutServer,
    _viewport,
)
from run_external_600_siliconflow import (
    DEFAULT_BASE_URL,
    _api_key,
    _extract_html,
    _load_env_file,
    _post_chat_completion,
)

from webmark.benchmark_baselines import evaluate_reference_fit_baselines
from webmark.prompt_factorial import (
    CONDITIONS,
    PAGE_TYPES,
    PROMPT_FACTORIAL_SCHEMA,
    FactorialPage,
    analyze_prompt_factorial,
    build_factorial_plan,
    plan_as_dicts,
    prompt_messages,
)
from webmark.release import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "webhumanbench_v1_prompt_factorial.json"
DEFAULT_SOURCE_RUN = ROOT / "results" / "webhumanbench_v1_prompt_factorial_source_run.json"
DEFAULT_ARTIFACT_ROOT = ROOT / "results" / "webhumanbench_v1_prompt_factorial_artifacts"
DEFAULT_RECOMPUTE_AUDIT = ROOT / "results" / "webhumanbench_v1_prompt_factorial_recompute_audit.json"
DEFAULT_MANIFEST = ROOT / "benchmark" / "webhumanbench_v1" / "webhumanbench_manifest.json"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_VIEWPORT = "390x844"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _artifact(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _validate_frozen_source_run(
    source_run: dict[str, Any],
    plan: list[FactorialPage],
    *,
    artifact_root: Path,
) -> None:
    if source_run.get("schema") != "webmark_prompt_factorial_source_run_v1":
        raise ValueError("source run has an unsupported schema")
    if source_run.get("status") != "complete":
        raise ValueError("frozen source run is not complete")
    rows = source_run.get("per_page")
    if not isinstance(rows, list):
        raise ValueError("source run requires a per_page array")
    config = source_run.get("config")
    if not isinstance(config, dict):
        raise ValueError("source run requires a config object")
    expected_config = {
        "model": plan[0].model,
        "blocks_per_type": len(plan) // (len(PAGE_TYPES) * len(CONDITIONS)),
        "planned_pages": len(plan),
        "seed": min(page.seed for page in plan),
    }
    for key, expected_value in expected_config.items():
        if config.get(key) != expected_value:
            raise ValueError(f"source-run config field {key} does not match the plan")
    expected = {page.page_id: page for page in plan}
    observed = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    if set(observed) != set(expected):
        raise ValueError("source-run page IDs do not match the deterministic factorial plan")
    for page_id, page in expected.items():
        row = observed[page_id]
        if row.get("status") != "ok":
            raise ValueError(f"source-run row {page_id} is not successful")
        for key, expected_value in page.__dict__.items():
            if row.get(key) != expected_value:
                raise ValueError(f"source-run row {page_id} changed plan field {key}")
        paths = _paths(page, artifact_root)
        checks = {
            "prompt_sha256": paths["prompt"],
            "generation_config_sha256": paths["config"],
            "raw_response_sha256": paths["raw_response"],
            "generated_html_sha256": paths["generated_html"],
        }
        for field, path in checks.items():
            if not path.is_file():
                raise ValueError(f"source-run artifact is missing: {path}")
            if row.get(field) != sha256_file(path):
                raise ValueError(f"source-run artifact hash mismatch: {path}")


def _validate_frozen_capture(
    result: dict[str, Any],
    plan: list[FactorialPage],
    *,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    if result.get("schema") != PROMPT_FACTORIAL_SCHEMA:
        raise ValueError("capture result has an unsupported schema")
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError("capture result requires a records array")
    expected_ids = {f"ai-{page.page_id}" for page in plan}
    observed_ids = {str(record.get("id")) for record in records if isinstance(record, dict)}
    if observed_ids != expected_ids:
        raise ValueError("capture IDs do not match the deterministic factorial plan")
    browser = str(result.get("capture", {}).get("browser", ""))
    if not browser:
        raise ValueError("capture result does not record a browser version")
    for record in records:
        provenance = record.get("provenance", {})
        if provenance.get("browser_version") != browser:
            raise ValueError(f"record {record.get('id')} has inconsistent browser provenance")
        artifacts = provenance.get("artifacts", {})
        for name, artifact in artifacts.items():
            path = artifact_root / str(artifact.get("path", ""))
            if not path.is_file():
                raise ValueError(f"capture artifact is missing: {path}")
            if artifact.get("sha256") != sha256_file(path):
                raise ValueError(f"capture artifact hash mismatch: {name} for {record.get('id')}")
    return records


def _paths(page: FactorialPage, artifact_root: Path) -> dict[str, Path]:
    return {
        "prompt": artifact_root / "prompts" / f"{page.page_id}.json",
        "config": artifact_root / "configs" / f"{page.page_id}.json",
        "raw_response": artifact_root / "raw_responses" / f"{page.page_id}.json",
        "generated_html": artifact_root / "generated_html" / f"{page.page_id}.html",
        "rendered_html": artifact_root / "rendered_html" / f"{page.page_id}.html",
        "screenshot": artifact_root / "screenshots" / f"{page.page_id}.png",
        "computed_features": artifact_root / "computed_features" / f"{page.page_id}.json",
    }


def _prepare_capture_output_paths(paths: dict[str, Path]) -> None:
    """Create capture-only directories without touching frozen source artifacts."""
    for name in ("rendered_html", "screenshot", "computed_features"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)


def _generation_config(
    page: FactorialPage,
    *,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_url": base_url,
        "model": page.model,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
        "experimental_condition": page.condition,
        "literal_css": page.literal_css,
        "human_style": page.human_style,
    }
    if page.model.startswith(("Qwen/Qwen3", "deepseek-ai/DeepSeek", "zai-org/GLM", "Pro/")):
        config["enable_thinking"] = False
    return config


def _generation_record(
    page: FactorialPage,
    paths: dict[str, Path],
    response: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    row = {
        **page.__dict__,
        "id": page.page_id,
        "status": status,
        "prompt_path": _relative(paths["prompt"]),
        "generation_config_path": _relative(paths["config"]),
        "raw_response_path": _relative(paths["raw_response"]),
        "initial_html_path": _relative(paths["generated_html"]),
        "prompt_sha256": sha256_file(paths["prompt"]),
        "generation_config_sha256": sha256_file(paths["config"]),
        "usage": response.get("usage", {}),
        "provider_created": response.get("created"),
        "provider_model": response.get("model"),
        "trace_id": response.get("_webmark_headers", {}).get("x-siliconcloud-trace-id", ""),
    }
    if status == "ok":
        row["raw_response_sha256"] = sha256_file(paths["raw_response"])
        row["generated_html_sha256"] = sha256_file(paths["generated_html"])
    if error:
        row["error"] = error
    return row


def _generate_one(
    page: FactorialPage,
    *,
    artifact_root: Path,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
) -> dict[str, Any]:
    paths = _paths(page, artifact_root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    messages = prompt_messages(page)
    _write_json(paths["prompt"], messages)
    _write_json(
        paths["config"],
        _generation_config(
            page,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
    response: dict[str, Any] = {}
    try:
        if paths["raw_response"].is_file() and paths["generated_html"].is_file():
            with paths["raw_response"].open(encoding="utf-8") as handle:
                response = json.load(handle)
            _extract_html(paths["generated_html"].read_text(encoding="utf-8"))
        else:
            response, headers = _post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=page.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                retries=retries,
                retry_sleep_s=2.0,
            )
            response["_webmark_headers"] = {
                "x-siliconcloud-trace-id": headers.get("x-siliconcloud-trace-id", "")
            }
            _write_json(paths["raw_response"], response)
            content = response["choices"][0]["message"]["content"]
            paths["generated_html"].write_text(_extract_html(content), encoding="utf-8")
        return _generation_record(page, paths, response, status="ok")
    except Exception as exc:
        return _generation_record(
            page,
            paths,
            response,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _source_run_payload(
    plan: list[FactorialPage],
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    blocks_per_type: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    artifact_root: Path,
    started_at: str,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["id"])
    return {
        "schema": "webmark_prompt_factorial_source_run_v1",
        "status": "complete" if len(ordered) == len(plan) and all(row["status"] == "ok" for row in ordered) else "partial",
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "config": {
            "base_url": base_url,
            "model": model,
            "blocks_per_type": blocks_per_type,
            "planned_pages": len(plan),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "artifact_root": _relative(artifact_root),
            "request_order": "deterministically_shuffled",
        },
        "plan": plan_as_dicts(plan),
        "per_page": ordered,
        "summary": {
            "planned": len(plan),
            "completed": sum(row["status"] == "ok" for row in ordered),
            "failed": sum(row["status"] != "ok" for row in ordered),
            "usage": dict(
                Counter(
                    {
                        key: sum(
                            int(row.get("usage", {}).get(key, 0))
                            for row in ordered
                            if isinstance(row.get("usage", {}).get(key, 0), int)
                        )
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                )
            ),
        },
        "note": (
            "The key is never stored. Provider responses, prompts, configurations, and extracted HTML "
            "are retained before browser scoring."
        ),
    }


def generate_factorial(
    plan: list[FactorialPage],
    *,
    source_run_output: Path,
    artifact_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    blocks_per_type: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    workers: int,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    request_order = list(plan)
    random.Random(seed).shuffle(request_order)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_one,
                page,
                artifact_root=artifact_root,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                retries=retries,
            ): page
            for page in request_order
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            payload = _source_run_payload(
                plan,
                rows,
                base_url=base_url,
                model=model,
                blocks_per_type=blocks_per_type,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
                artifact_root=artifact_root,
                started_at=started_at,
            )
            _write_json(source_run_output, payload)
            if completed % 10 == 0 or completed == len(plan):
                print(
                    f"generation [{completed}/{len(plan)}] "
                    f"ok={payload['summary']['completed']} failed={payload['summary']['failed']}"
                )
    return _source_run_payload(
        plan,
        rows,
        base_url=base_url,
        model=model,
        blocks_per_type=blocks_per_type,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        artifact_root=artifact_root,
        started_at=started_at,
    )


def capture_factorial(
    source_run: dict[str, Any],
    *,
    artifact_root: Path,
    viewport: str,
    chrome: Path,
    settle_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    successful = [row for row in source_run["per_page"] if row["status"] == "ok"]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with _StaticCheckoutServer(artifact_root) as server:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError('Install browser dependencies with: pip install -e ".[browser]"') from exc
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=str(chrome) if chrome.exists() else None,
            )
            browser_version = browser.version
            try:
                for index, row in enumerate(successful, start=1):
                    page_id = str(row["id"])
                    page = FactorialPage(
                        page_id=page_id,
                        block_id=str(row["block_id"]),
                        page_type=str(row["page_type"]),
                        condition=str(row["condition"]),
                        literal_css=bool(row["literal_css"]),
                        human_style=bool(row["human_style"]),
                        model=str(row["model"]),
                        index=int(row["index"]),
                        seed=int(row["seed"]),
                        scenario=str(row["scenario"]),
                    )
                    paths = _paths(page, artifact_root)
                    _prepare_capture_output_paths(paths)
                    context = browser.new_context(
                        locale="en-US",
                        timezone_id="UTC",
                        color_scheme="light",
                        reduced_motion="reduce",
                        device_scale_factor=1,
                    )
                    blocked_requests: list[str] = []
                    context.route("**/*", _local_only_route(blocked_requests))
                    try:
                        browser_page = context.new_page()
                        browser_page.set_viewport_size(_viewport(viewport))
                        target = f"{server.url}/generated_html/{page_id}.html"
                        response = browser_page.goto(
                            target, wait_until="domcontentloaded", timeout=30_000
                        )
                        browser_page.evaluate("document.fonts.ready")
                        browser_page.wait_for_timeout(settle_ms)
                        if blocked_requests:
                            raise RuntimeError(f"blocked external request: {blocked_requests[0]}")
                        if response is None or not 200 <= response.status < 400:
                            raise RuntimeError("local render did not return a successful response")
                        features = browser_page.evaluate(_computed_feature_script())
                        required = ("typography", "spacing", "grid", "color", "saturation")
                        if not all(features.get(name) for name in required):
                            raise RuntimeError("incomplete computed feature vector")
                        paths["rendered_html"].write_text(
                            browser_page.content(), encoding="utf-8"
                        )
                        browser_page.screenshot(path=str(paths["screenshot"]), full_page=False)
                        _write_json(paths["computed_features"], features)
                        artifacts = {
                            name: _artifact(path, artifact_root)
                            for name, path in paths.items()
                        }
                        records.append(
                            {
                                "id": f"ai-{page_id}",
                                "source": "ai",
                                "split": "test",
                                "group_id": f"generation-{page_id}",
                                "leakage_group_id": f"prompt-factorial-{page.block_id}",
                                "page_type": page.page_type,
                                "viewport": viewport,
                                "model_id": page.model,
                                "features": features,
                                "condition": page.condition,
                                "block_id": page.block_id,
                                "provenance": {
                                    "provider": source_run["config"]["base_url"],
                                    "model_id": page.model,
                                    "prompt_condition": page.condition,
                                    "literal_css": page.literal_css,
                                    "human_style": page.human_style,
                                    "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
                                    "browser_version": browser_version,
                                    "artifacts": artifacts,
                                },
                            }
                        )
                    except Exception as exc:
                        failures.append({"page_id": page_id, "error": str(exc)})
                    finally:
                        context.close()
                    if index % 20 == 0 or index == len(successful):
                        print(
                            f"capture [{index}/{len(successful)}] "
                            f"ok={len(records)} failed={len(failures)}"
                        )
            finally:
                browser.close()
    return records, failures, browser_version


def _factorial_manifest(
    benchmark_manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    metadata = copy.deepcopy(benchmark_manifest["metadata"])
    metadata["version"] = f"{metadata['version']}-prompt-factorial-v1"
    historical = [
        copy.deepcopy(record)
        for record in benchmark_manifest["records"]
        if record.get("source") == "human"
    ]
    scoring_records = [
        {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key
            in {
                "id",
                "source",
                "split",
                "group_id",
                "leakage_group_id",
                "page_type",
                "viewport",
                "model_id",
                "features",
            }
        }
        for record in records
    ]
    return {"schema": benchmark_manifest["schema"], "metadata": metadata, "records": historical + scoring_records}


def _recompute_analysis(
    benchmark_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    plan: list[FactorialPage],
    *,
    n_bootstrap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment_manifest = _factorial_manifest(benchmark_manifest, records)
    baseline_result = evaluate_reference_fit_baselines(
        experiment_manifest,
        n_resamples=n_bootstrap,
        seed=42,
        min_groups_for_ci=10,
    )
    analysis = analyze_prompt_factorial(
        baseline_result,
        records,
        plan_as_dicts(plan),
        n_resamples=n_bootstrap,
        seed=91,
    )
    return baseline_result, analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-run-output", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--recompute-audit-output", type=Path, default=DEFAULT_RECOMPUTE_AUDIT)
    parser.add_argument(
        "--recomputed-result-output",
        type=Path,
        help="with --analysis-only, write a copy containing the recomputed statistics",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocks-per-type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout-s", type=int, default=150)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--viewport", default=DEFAULT_VIEWPORT)
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--settle-ms", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-source-run",
        action="store_true",
        help="reuse and hash-validate the frozen source run without an API key",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="hash-validate the frozen capture and recompute analysis without browser or API calls",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.n_bootstrap <= 0:
        raise ValueError("workers and bootstrap count must be positive")
    _viewport(args.viewport)
    plan = build_factorial_plan(
        model=args.model,
        blocks_per_type=args.blocks_per_type,
        seed=args.seed,
    )
    if args.dry_run:
        _write_json(
            args.source_run_output,
            {
                "schema": "webmark_prompt_factorial_source_run_v1",
                "status": "dry_run",
                "config": {
                    "base_url": args.base_url,
                    "model": args.model,
                    "blocks_per_type": args.blocks_per_type,
                    "planned_pages": len(plan),
                    "seed": args.seed,
                },
                "plan": plan_as_dicts(plan),
            },
        )
        print(f"planned {len(plan)} pages")
        return 0

    if args.analysis_only:
        frozen_result = _load_json(args.output)
        source_run = _load_json(args.source_run_output)
        _validate_frozen_source_run(source_run, plan, artifact_root=args.artifact_root)
        records = _validate_frozen_capture(
            frozen_result,
            plan,
            artifact_root=args.artifact_root,
        )
        benchmark_manifest = _load_json(args.manifest)
        baseline_result, analysis = _recompute_analysis(
            benchmark_manifest,
            records,
            plan,
            n_bootstrap=args.n_bootstrap,
        )
        stored_baseline_hash = canonical_json_sha256(frozen_result["reference_fit_baselines"])
        recomputed_baseline_hash = canonical_json_sha256(baseline_result)
        stored_analysis_hash = canonical_json_sha256(frozen_result["analysis"])
        recomputed_analysis_hash = canonical_json_sha256(analysis)
        matches = (
            stored_baseline_hash == recomputed_baseline_hash
            and stored_analysis_hash == recomputed_analysis_hash
        )
        audit = {
            "schema": "webmark_prompt_factorial_recompute_audit_v1",
            "status": "pass" if matches else "mismatch",
            "inputs": {
                "result": {"path": _relative(args.output), "sha256": sha256_file(args.output)},
                "source_run": {
                    "path": _relative(args.source_run_output),
                    "sha256": sha256_file(args.source_run_output),
                },
                "benchmark_manifest": {
                    "path": _relative(args.manifest),
                    "sha256": sha256_file(args.manifest),
                },
                "artifact_root": _relative(args.artifact_root),
                "validated_capture_records": len(records),
            },
            "comparison": {
                "stored_baseline_sha256": stored_baseline_hash,
                "recomputed_baseline_sha256": recomputed_baseline_hash,
                "stored_analysis_sha256": stored_analysis_hash,
                "recomputed_analysis_sha256": recomputed_analysis_hash,
            },
            "determinism_boundary": (
                "The audit compares deterministic baseline and analysis objects. Provider timestamps, "
                "request latency, and total run elapsed time are intentionally excluded."
            ),
        }
        _write_json(args.recompute_audit_output, audit)
        if args.recomputed_result_output is not None:
            refreshed = copy.deepcopy(frozen_result)
            refreshed["reference_fit_baselines"] = baseline_result
            refreshed["analysis"] = analysis
            _write_json(args.recomputed_result_output, refreshed)
        print(f"prompt-factorial offline recomputation: {audit['status']}")
        return 0 if matches or args.recomputed_result_output is not None else 1

    start = time.time()
    if args.reuse_source_run:
        source_run = _load_json(args.source_run_output)
        _validate_frozen_source_run(source_run, plan, artifact_root=args.artifact_root)
    else:
        _load_env_file(args.env_file)
        api_key = _api_key(args.api_key_env)
        source_run = generate_factorial(
            plan,
            source_run_output=args.source_run_output,
            artifact_root=args.artifact_root,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            blocks_per_type=args.blocks_per_type,
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            retries=args.retries,
            workers=args.workers,
        )
        _write_json(args.source_run_output, source_run)
    records, capture_failures, browser_version = capture_factorial(
        source_run,
        artifact_root=args.artifact_root,
        viewport=args.viewport,
        chrome=args.chrome,
        settle_ms=args.settle_ms,
    )
    if capture_failures or len(records) != len(plan):
        failure_audit = {
            "schema": "webmark_prompt_factorial_capture_failure_audit_v1",
            "status": "incomplete",
            "source_run": {
                "path": _relative(args.source_run_output),
                "sha256": sha256_file(args.source_run_output),
            },
            "artifact_root": _relative(args.artifact_root),
            "capture": {
                "viewport": args.viewport,
                "browser": browser_version,
                "settle_ms": args.settle_ms,
                "n_records": len(records),
                "n_excluded": len(capture_failures),
                "exclusions": capture_failures,
            },
            "claim_boundary": (
                "This is a failed capture audit, not a benchmark result. "
                "It must not be cited as a completed prompt-factorial experiment."
            ),
        }
        failure_path = args.output.with_name(f"{args.output.stem}_capture_failures.json")
        _write_json(failure_path, failure_audit)
        raise RuntimeError(
            "prompt-factorial capture is incomplete; "
            f"wrote failure audit to {failure_path}"
        )
    benchmark_manifest = _load_json(args.manifest)
    baseline_result, analysis = _recompute_analysis(
        benchmark_manifest,
        records,
        plan,
        n_bootstrap=args.n_bootstrap,
    )
    output = {
        "schema": PROMPT_FACTORIAL_SCHEMA,
        "status": (
            "complete"
            if source_run["status"] == "complete"
            and not capture_failures
            and len(records) == len(plan)
            else "completed_with_exclusions"
        ),
        "source_run": {
            "path": _relative(args.source_run_output),
            "sha256": sha256_file(args.source_run_output),
            "summary": source_run["summary"],
        },
        "benchmark_manifest_sha256": canonical_json_sha256(benchmark_manifest),
        "artifact_root": _relative(args.artifact_root),
        "capture": {
            "viewport": args.viewport,
            "browser": browser_version,
            "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
            "settle_ms": args.settle_ms,
            "n_records": len(records),
            "n_excluded": len(capture_failures),
            "exclusions": capture_failures,
        },
        "analysis": analysis,
        "reference_fit_baselines": baseline_result,
        "records": records,
        "elapsed_s": time.time() - start,
    }
    _write_json(args.output, output)
    profile = analysis["baselines"]["profile_l2_w1"]
    print(f"prompt factorial: {output['status']}; captured {len(records)}/{len(plan)}")
    for condition, endpoint in profile["condition_type_macro_auroc"].items():
        print(f"  {condition}: type-macro AUROC={endpoint['point_estimate']:.4f}")
    for factor in ("literal_css", "human_style"):
        effect = profile["paired_factor_effects"]["effects"][factor]
        print(
            f"  {factor}: closer-fit rate={effect['equal_page_type_paired_closer_fit_rate']:.4f} "
            f"CI=[{effect['ci_95'][0]:.4f}, {effect['ci_95'][1]:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
