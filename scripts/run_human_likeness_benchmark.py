#!/usr/bin/env python3
"""Run the WebHumanBench reference-relative design human-likeness protocol.

The input manifest must satisfy the leakage and provenance contract documented
in ``docs/HUMAN_LIKENESS_BENCHMARK.md``. The bundled example is synthetic and
tests only the executable schema; it is not a benchmark result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.human_likeness import evaluate_human_likeness_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON benchmark manifest")
    parser.add_argument("--output", type=Path, required=True, help="JSON evaluation output")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    result = evaluate_human_likeness_benchmark(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    test = result["test"]
    print(f"benchmark: {result['benchmark']} ({result['version']})")
    print(f"test rows: {test['n_rows']}")
    print(f"human-vs-AI AUROC: {test['human_vs_ai_auroc']:.4f}")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
