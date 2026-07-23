#!/usr/bin/env python3
"""Audit temporal, capture-intervention, and generation-prompt boundaries for v1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.benchmark_protocol_audit import build_protocol_audit
from webmark.release import sha256_file


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--capture-ledger", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--ai-source-run-ledger", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_paths = {
        "source_manifest": args.source_manifest,
        "capture_ledger": args.capture_ledger,
        "benchmark_manifest": args.benchmark_manifest,
        "ai_source_run_ledger": args.ai_source_run_ledger,
    }
    result = build_protocol_audit(
        _load(args.source_manifest),
        _load(args.capture_ledger),
        _load(args.benchmark_manifest),
        _load(args.ai_source_run_ledger),
        capture_root=args.capture_root,
    )
    result["inputs"] = {
        name: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    reference = result["reference_capture_interventions"]
    prompts = result["generated_prompt_protocol"]
    print(
        "reference sources/assets: "
        f"{reference['n_sources']}/{reference['vendored_asset_count']} "
        f"({reference['sources_with_vendored_assets']} sources with vendored assets)"
    )
    print(
        "audited prompts/templates: "
        f"{prompts['audited_retained_prompt_artifacts']}/{prompts['normalized_user_templates']}"
    )
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
