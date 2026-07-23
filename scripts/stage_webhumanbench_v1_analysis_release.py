#!/usr/bin/env python3
"""Stage the fixed WebHumanBench v1 anonymous analysis-only release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.release_bundle import stage_webhumanbench_release  # noqa: E402, I001


RELEASE_DIRECTORY = "webhumanbench_v1"
DEFAULT_STAGE_ROOT = Path(__file__).resolve().parents[2] / "release" / (
    "webhumanbench-v1.0.0-rc5-anonymous-analysis"
)
CORE_BENCHMARK_FILES = (
    "source_manifest.json",
    "capture_ledger.json",
    "split_assignments.json",
    "ai_records.json",
    "ai_source_run_ledger.json",
    "ai_exclusion_ledger.json",
    "webhumanbench_manifest.json",
    "corpus_assembly_receipt.json",
    "capture_rebind_receipt.json",
    "source_receipts.json",
)
RESULT_FILES = (
    "webhumanbench_v1_public_release_audit.json",
    "webhumanbench_v1_human_likeness_baseline.json",
    "webhumanbench_v1_reference_baselines.json",
    "webhumanbench_v1_reference_design_profile.json",
    "webhumanbench_v1_sensitivity_diagnostics.json",
    "webhumanbench_v1_protocol_audit.json",
    "webhumanbench_v1_readiness.json",
    "webhumanbench_v1_render_screen_r5.json",
    "webhumanbench_v1_prompt_factorial.json",
    "webhumanbench_v1_prompt_factorial_source_run.json",
    "webhumanbench_v1_prompt_factorial_browser_150_0_7871_129.json",
    "webhumanbench_v1_prompt_factorial_browser_patch_audit.json",
    "webhumanbench_v1_prompt_factorial_browser_150_0_7871_115_recompute_audit.json",
    "webhumanbench_v1_uiclip_image_embeddings.json",
    "webhumanbench_v1_visual_embedding_baselines.json",
)
DOCUMENT_FILES = (
    "DATA_CARD.md",
    "THIRD_PARTY_ARTIFACT_NOTICES.md",
    "docs/FORMAL_BENCHMARK_GATE.md",
    "docs/REPRODUCING.md",
    "docs/WEBHUMANBENCH_V1_RELEASE.md",
    "docs/WEBHUMANBENCH_V1_REFERENCE_DESIGN_PROFILE.md",
    "benchmark/README.md",
)
RUNTIME_SCRIPT_FILES = (
    "scripts/analyze_webhumanbench_reference_profile.py",
    "scripts/assemble_webhumanbench_v1_corpus.py",
    "scripts/audit_benchmark_readiness.py",
    "scripts/audit_public_release.py",
    "scripts/audit_reference_render_integrity.py",
    "scripts/build_webhumanbench_v1_ai_ledgers.py",
    "scripts/capture_open_mobile_reference.py",
    "scripts/capture_vendored_static_snapshots.py",
    "scripts/compare_prompt_factorial_browser_patches.py",
    "scripts/materialize_pinned_entrypoint_closures.py",
    "scripts/preflight_public_release.py",
    "scripts/rebind_webhumanbench_v1_capture_ledger.py",
    "scripts/run_external_600_siliconflow.py",
    "scripts/run_human_likeness_benchmark.py",
    "scripts/run_prompt_factorial.py",
    "scripts/run_reference_fit_baselines.py",
    "scripts/run_visual_embedding_baseline.py",
    "scripts/run_webhumanbench_v1_diagnostics.py",
    "scripts/run_webhumanbench_v1_protocol_audit.py",
    "scripts/stage_webhumanbench_release_bundle.py",
    "scripts/stage_webhumanbench_v1_analysis_release.py",
)
TEST_FILES = (
    "tests/test_assemble_webhumanbench_v1_corpus.py",
    "tests/test_benchmark_baselines.py",
    "tests/test_benchmark_diagnostics.py",
    "tests/test_benchmark_protocol_audit.py",
    "tests/test_benchmark_readiness.py",
    "tests/test_design_profile.py",
    "tests/test_human_likeness.py",
    "tests/test_prompt_factorial.py",
    "tests/test_prompt_factorial_browser_patch.py",
    "tests/test_visual_embedding_baseline.py",
    "tests/test_preflight_public_release.py",
    "tests/test_public_release.py",
    "tests/test_rebind_webhumanbench_v1_capture_ledger.py",
    "tests/test_reference_render_integrity.py",
    "tests/test_release_bundle.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="replace an existing staged tree after rebuilding its manifest",
    )
    args = parser.parse_args()

    result = stage_webhumanbench_release(
        Path(__file__).resolve().parents[1],
        args.stage_root,
        release_directory=RELEASE_DIRECTORY,
        result_files=RESULT_FILES,
        document_files=DOCUMENT_FILES,
        runtime_script_files=RUNTIME_SCRIPT_FILES,
        test_files=TEST_FILES,
        closure_manifest_directory="closure_manifests",
        core_benchmark_files=CORE_BENCHMARK_FILES,
        include_capture_artifacts=False,
        resume=args.refresh,
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
