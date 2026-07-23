#!/usr/bin/env python3
"""Run train-only reference-fit baselines on a WebHumanBench manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.benchmark_baselines import evaluate_reference_fit_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-groups-for-ci", type=int, default=10)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    result = evaluate_reference_fit_baselines(
        manifest,
        n_resamples=args.n_bootstrap,
        seed=args.seed,
        min_groups_for_ci=args.min_groups_for_ci,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for name, baseline in result["baselines"].items():
        primary = baseline["type_macro"]
        pooled = baseline["overall"]
        print(
            f"{name}: type-macro AUROC={primary['point_estimate']:.4f} "
            f"({primary['status']}), pooled compatibility={pooled['point_estimate']:.4f}"
        )
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
