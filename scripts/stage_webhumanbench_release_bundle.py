#!/usr/bin/env python3
"""Stage a minimal, hash-indexed WebHumanBench release bundle.

Only artifacts reachable from the frozen capture and AI manifests are copied.
Candidate queues, caches, dependency directories, and AppleDouble sidecars are
excluded before the separate public-release preflight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.release_bundle import stage_webhumanbench_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source code directory containing benchmark/<release-directory>",
    )
    parser.add_argument("--release-directory", required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--result-file", action="append", required=True)
    parser.add_argument("--document-file", action="append", required=True)
    parser.add_argument("--runtime-script", action="append", required=True)
    parser.add_argument(
        "--test-file",
        action="append",
        default=[],
        help="code-root-relative release-facing test file to include",
    )
    parser.add_argument("--result-directory", action="append", default=[])
    parser.add_argument("--benchmark-file", action="append", default=[])
    parser.add_argument("--benchmark-directory", action="append", default=[])
    parser.add_argument(
        "--closure-manifest-directory",
        help="copy only closure manifests referenced by the frozen source receipts",
    )
    parser.add_argument(
        "--core-benchmark-file",
        action="append",
        help="version-specific core file; when omitted, uses the legacy v0.2 core set",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "omit raw captures, vendored assets, generated HTML, provider responses, and screenshots; "
            "retain manifests, derived features, results, code, and tests"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = stage_webhumanbench_release(
        args.code_root,
        args.stage_root,
        release_directory=args.release_directory,
        result_files=args.result_file,
        document_files=args.document_file,
        runtime_script_files=args.runtime_script,
        test_files=args.test_file,
        result_directories=args.result_directory,
        benchmark_files=args.benchmark_file,
        benchmark_directories=args.benchmark_directory,
        closure_manifest_directory=args.closure_manifest_directory,
        core_benchmark_files=args.core_benchmark_file,
        include_capture_artifacts=not args.analysis_only,
        resume=args.resume,
        refresh=args.refresh,
    )
    print(f"staged: {args.stage_root}")
    print(f"version/revision: {result['version']} / {result['release_revision']}")
    print(f"referenced artifacts: {result['referenced_capture_artifact_count']}")
    print(f"included artifacts: {result['included_capture_artifact_count']}")
    print(f"distribution profile: {result['distribution_profile']}")
    print(f"payload files/bytes: {result['payload_file_count']} / {result['payload_bytes']}")
    print(f"payload tree sha256: {result['payload_tree_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
