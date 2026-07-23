#!/usr/bin/env python3
"""Describe the released fixed-commit WebHumanBench reference cohort.

The command is intentionally descriptive.  It derives typography, spacing,
layout-phase, palette, and saturation summaries from the already captured
reference corpus, but it does not infer individual authorship or assign an
aesthetic score.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.design_profile import (  # noqa: E402
    build_reference_design_profile,
    reference_design_profile_markdown,
)


def _load_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--capture-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    report = build_reference_design_profile(
        _load_object(args.source_manifest), _load_object(args.capture_ledger)
    )
    report["created_at"] = datetime.now(UTC).isoformat()
    report["source_manifest_path"] = str(args.source_manifest)
    report["capture_ledger_path"] = str(args.capture_ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(reference_design_profile_markdown(report), encoding="utf-8")
    print(
        f"profiled {report['source_groups']} fixed-commit source groups across "
        f"{report['captures']} retained mobile captures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
