# Reproducing WebHumanBench v1.0.0

Run commands from the repository root (the staged package's `code/`
directory). The local v1.0.0 package is the source of record for paper claims.
Historical controller experiments and v0.2 artifacts are not v1 benchmark
evidence. Keep recomputed reports outside the checkout so the audited upload
tree remains unchanged:

```bash
mkdir -p ../webhumanbench-recomputed
```

## Environment

```bash
python -m pip install -e ".[dev,browser]"
python -m pytest -q
python -m ruff check src scripts tests
```

Frozen captures use Chromium `150.0.7871.115`. The analysis-only package can
verify its included ledgers and feature-bearing manifest. Byte-level capture
verification and fresh capture require the restricted raw-artifact bundle, the
declared browser, fixed-commit source access, and availability of recorded
external asset URLs; new bytes define a successor capture rather than a
byte-identical reproduction.

## Verify Artifact Integrity

The script name retains `public` for compatibility, but this command audits the
local candidate and does not claim external publication. In the analysis-only
profile, the expected report states `artifact hashes checked: False`, because
the raw artifacts are intentionally absent.

```bash
python scripts/audit_public_release.py \
  --source-manifest benchmark/webhumanbench_v1/source_manifest.json \
  --capture-ledger benchmark/webhumanbench_v1/capture_ledger.json \
  --benchmark-manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_release_candidate_audit_recomputed.json
```

Expected included-ledger counts are 32 sources, 64 reference captures, 590
generated records, and 622 scoring records.

With authorized access to the restricted raw-artifact bundle, run the separate
render-integrity screen against that bundle:

```bash
python scripts/audit_reference_render_integrity.py \
  --source-manifest benchmark/webhumanbench_v1/source_manifest.json \
  --capture-ledger benchmark/webhumanbench_v1/capture_ledger.json \
  --artifact-root /path/to/restricted/webhumanbench_v1/captures \
  --output ../webhumanbench-recomputed/webhumanbench_v1_render_screen_recomputed.json
```

The frozen candidate reports all 32 reference sources passing both mobile
captures; a fresh restricted-bundle run is a provenance check, not a new
benchmark score.

## Rebuild the 600-Row AI Ledgers

```bash
python scripts/build_webhumanbench_v1_ai_ledgers.py \
  --source-run results/external_600_siliconflow.json \
  --ai-archive results/webhumanbench_v0_2_ai_baseline_archive_r1.json \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --source-ledger-output benchmark/webhumanbench_v1/ai_source_run_ledger.json \
  --exclusion-ledger-output benchmark/webhumanbench_v1/ai_exclusion_ledger.json
```

The command must report 600 planned, 590 retained, and 10 excluded. It rejects
duplicate IDs, score-dependent selection, inconsistent model/type metadata, or
any retained/excluded set that does not close the source run.

## Recompute Baselines and Profiles

```bash
python scripts/run_human_likeness_benchmark.py \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_human_likeness_baseline_recomputed.json

python scripts/run_reference_fit_baselines.py \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --n-bootstrap 2000 --seed 42 --min-groups-for-ci 10 \
  --output ../webhumanbench-recomputed/webhumanbench_v1_reference_baselines_recomputed.json

python scripts/analyze_webhumanbench_reference_profile.py \
  --source-manifest benchmark/webhumanbench_v1/source_manifest.json \
  --capture-ledger benchmark/webhumanbench_v1/capture_ledger.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_reference_design_profile_recomputed.json \
  --markdown-output ../webhumanbench-recomputed/WEBHUMANBENCH_V1_REFERENCE_DESIGN_PROFILE_RECOMPUTED.md

python scripts/audit_benchmark_readiness.py \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_readiness_recomputed.json
```

The readiness status must remain `pilot_only` for v1.

## Recompute Analysis-Only Sensitivity Diagnostics

```bash
python scripts/run_webhumanbench_v1_diagnostics.py \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --reference-baselines results/webhumanbench_v1_reference_baselines.json \
  --scored-baseline results/webhumanbench_v1_human_likeness_baseline.json \
  --design-profile results/webhumanbench_v1_reference_design_profile.json \
  --ai-exclusion-ledger benchmark/webhumanbench_v1/ai_exclusion_ledger.json \
  --ai-source-run-ledger benchmark/webhumanbench_v1/ai_source_run_ledger.json \
  --n-splits 100 --n-bootstrap 2000 --seed 2027 \
  --bootstrap-seed 42 \
  --output ../webhumanbench-recomputed/webhumanbench_v1_sensitivity_diagnostics_recomputed.json
```

The protocol audit requires the restricted capture bundle because it verifies
capture-time intervention artifacts. With authorized access, run it with
`--capture-root /path/to/restricted/webhumanbench_v1/captures` and write the
report outside the checkout.

## Frozen Prompt and Visual Controls

The package publishes the prompt-factorial ledgers and result, but not the raw
provider responses, generated HTML, screenshots, or artifact directory. It
therefore cannot replay or independently hash-validate that factorial from the
analysis-only release. With authorized access to the restricted artifact bundle,
the following command performs the no-network replay; it does not require an
API key:

```bash
python scripts/run_prompt_factorial.py \
  --analysis-only \
  --source-run-output results/webhumanbench_v1_prompt_factorial_source_run.json \
  --artifact-root /path/to/restricted/webhumanbench_v1_prompt_factorial_artifacts \
  --output ../webhumanbench-recomputed/webhumanbench_v1_prompt_factorial.json \
  --recompute-audit-output ../webhumanbench-recomputed/webhumanbench_v1_prompt_factorial_recomputed_audit.json

python scripts/compare_prompt_factorial_browser_patches.py \
  --left results/webhumanbench_v1_prompt_factorial_browser_150_0_7871_129.json \
  --right ../webhumanbench-recomputed/webhumanbench_v1_prompt_factorial.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_prompt_factorial_browser_patch_audit_recomputed.json
```

The cached-embedding UIClip control scores can be recomputed from the
analysis-only release without raw screenshots or model weights:

```bash
python scripts/run_visual_embedding_baseline.py \
  --analysis-only --n-bootstrap 2000 --n-splits 100 \
  --embedding-output results/webhumanbench_v1_uiclip_image_embeddings.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_visual_embedding_baselines_recomputed.json
```

The UIClip command validates 622 published screenshot-digest bindings, group
metadata, and cached 512-dimension embeddings before scoring. It does not
extract embeddings, recompute excluded screenshot bytes, or verify model
weights. It is an image-only representation control, not a human-rating,
authorship, or preference experiment.

The primary endpoint is equal-page-type macro AUROC with within-type bootstrap
resampling. The diagnostic artifact includes 100 source resplits,
leave-one-reference-out coverage over all 32 historical sources, score
aggregation, equal-model and leave-one-model-out sensitivity, seven model
strata, feature ablations, measurement-artifact checks, paired viewport
stability, and the 590/600 exclusion audit. The protocol artifact verifies all
retained prompt hashes and quantifies capture-time vendoring. The
`human_likeness` script and result names are retained only for schema
compatibility; the reported endpoint is reference-relative fit, not visual
authorship.

## Rebuild Reference Captures

Do not overwrite v1. Write fresh outputs to a separate directory:

```bash
python scripts/materialize_pinned_entrypoint_closures.py \
  --manifest benchmark/webhumanbench_v1/source_manifest.json \
  --checkout-root /tmp/webhumanbench_v1_closures \
  --closure-manifest-root /tmp/webhumanbench_v1_closure_manifests \
  --output /tmp/webhumanbench_v1_source_receipts.json

python scripts/capture_vendored_static_snapshots.py \
  --manifest benchmark/webhumanbench_v1/source_manifest.json \
  --checkout-root /tmp/webhumanbench_v1_closures \
  --checkout-receipts /tmp/webhumanbench_v1_source_receipts.json \
  --artifact-root /tmp/webhumanbench_v1_captures \
  --output /tmp/webhumanbench_v1_capture_ledger.json \
  --chrome '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
```

Compare the new receipts before promoting a successor. External asset bytes are
not guaranteed to match the 2026 frozen candidate.

## Stage the Candidate

The staging command copies only declared files. The anonymous analysis profile
keeps frozen summaries, browser-patch and image-embedding results, and the core
ledgers, but excludes raw prompt-factorial artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/stage_webhumanbench_v1_analysis_release.py --refresh

python scripts/preflight_public_release.py \
  --release-root ../release/webhumanbench-v1.0.0-rc5-anonymous-analysis \
  --output ../webhumanbench-recomputed/webhumanbench_v1_release_preflight_recomputed.json
```

The staged tree contains the release-facing tests needed by its own `pytest`
command. With `--analysis-only`, it intentionally omits raw captures, vendor
assets, generated HTML, raw provider responses, and screenshots while retaining
the feature-bearing manifest and all corresponding hashes. Preflight rejects
caches, AppleDouble metadata, credentials, symlinks, and oversized undeclared
files. Before upload, rerun the manifest-only artifact audit against the staged
tree and add a real repository identifier. Do not add a fabricated URL or
release date.

## Boundaries

Reproduction does not resolve prompt, content, asset, provider, encoder, or
source-selection confounding. Manual or LLM visual ratings are not benchmark
evidence because they answer a different subjective construct.
