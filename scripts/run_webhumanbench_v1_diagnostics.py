#!/usr/bin/env python3
"""Run frozen-data sensitivity diagnostics for WebHumanBench v1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.benchmark_diagnostics import build_v1_diagnostics
from webmark.release import sha256_file


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-baselines", type=Path, required=True)
    parser.add_argument("--scored-baseline", type=Path, required=True)
    parser.add_argument("--design-profile", type=Path, required=True)
    parser.add_argument("--ai-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--ai-source-run-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    input_paths = {
        "manifest": args.manifest,
        "reference_baselines": args.reference_baselines,
        "scored_baseline": args.scored_baseline,
        "design_profile": args.design_profile,
        "ai_exclusion_ledger": args.ai_exclusion_ledger,
        "ai_source_run_ledger": args.ai_source_run_ledger,
    }
    result = build_v1_diagnostics(
        _load(args.manifest),
        _load(args.reference_baselines),
        _load(args.scored_baseline),
        _load(args.design_profile),
        _load(args.ai_exclusion_ledger),
        _load(args.ai_source_run_ledger),
        n_alternative_splits=args.n_splits,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["inputs"] = {
        name: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"alternative source splits: {args.n_splits}")
    for name, summary in result["source_split_sensitivity"]["baseline_auroc_summary"].items():
        print(
            f"{name}: median={summary['median']:.4f}, "
            f"IQR=[{summary['q1']:.4f}, {summary['q3']:.4f}], "
            f"range=[{summary['min']:.4f}, {summary['max']:.4f}]"
        )
    print("leave-one-reference-out sourcewise coverage:")
    for name, baseline in result["leave_one_reference_out_coverage"]["baselines"].items():
        summary = baseline["summary"]
        print(
            f"{name}: mean={summary['mean']:.4f}, "
            f"95% CI=[{summary['ci_95'][0]:.4f}, {summary['ci_95'][1]:.4f}], "
            f"range=[{summary['min']:.4f}, {summary['max']:.4f}]"
        )
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
