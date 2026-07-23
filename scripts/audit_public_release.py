#!/usr/bin/env python3
"""Fail closed unless a public WebHumanBench release is cross-artifact complete.

This validates the source manifest, capture ledger, benchmark manifest, and,
when supplied, every saved HTML/screenshot digest. Passing it is a release
integrity result only. It does not establish human authorship, preference,
accessibility, or benchmark effectiveness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.release import validate_public_release


def _load_json(path: Path) -> dict[str, object]:
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
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="optional capture-artifact root; when supplied every HTML/PNG SHA-256 is recomputed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate_public_release(
        _load_json(args.source_manifest),
        _load_json(args.capture_ledger),
        _load_json(args.benchmark_manifest),
        artifact_root=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("public release integrity: passed")
    print(f"sources/captures/records: {result['n_sources']}/{result['n_captures']}/{result['n_benchmark_records']}")
    print(f"artifact hashes checked: {result['artifact_hashes_checked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
