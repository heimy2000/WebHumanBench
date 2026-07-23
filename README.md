# WebHumanBench

Reference implementation and v1.0.0-rc5 anonymous analysis-release candidate for the paper
*WebHumanBench: An Auditable Benchmark Pilot for Reference-Relative Web Design Fit*.

## Status

`benchmark/webhumanbench_v1/` is a locally frozen, hash-audited candidate.
The staged `1.0.0-rc5` package is an **analysis-only** anonymous release: it
contains code, ledgers, derived feature records, results, and tests, while raw
captures and other third-party/provider materials are deliberately excluded.
It must not be described as externally released until a real anonymous
repository points to that exact staged tree.

| Item | V1.0.0 candidate value |
| --- | --- |
| Historical-reference groups | 32 fixed-commit open-source sources |
| Reference captures | 64 local Chromium captures at `390x844` and `430x932` |
| Reference split | 16 train / 6 dev / 10 test source groups |
| Generated source run | 600 completed responses from 7 recorded model IDs |
| Generated retained/excluded | 590 / 10 under a score-independent closure rule |
| Scoring records | 622 at `390x844` |
| Formal readiness | `pilot_only` |

The benchmark measures fit to a declared browser-feature reference cohort. It
does not infer authorship, aesthetic quality, preference, usability, or
accessibility.

## Decisive Diagnostics

Five train-only frozen-split equal-page-type macro AUROCs range from `0.3790`
to `0.8426`, but these numbers are unstable and strongly feature-dependent:

- Profile L2+W1 over 100 alternative source splits: median `0.6234`, IQR
  `[0.5470, 0.6933]`, range `[0.4171, 0.8541]`.
- Profile leave-one-reference-out coverage over all 32 sources: mean `0.6582`,
  95% stratified source-bootstrap CI `[0.5666, 0.7433]`, range `[0, 1]`;
  dashboard mean `0.155` despite frozen dashboard AUROC `1.0`.
- Profile aggregation: equal-type macro `0.7910`, pair-weighted within-type
  `0.7086`, and pooled cross-type compatibility value `0.6934`.
- Equal-model Profile mean `0.7881`; leave-one-model-out range
  `[0.7731, 0.8200]`.
- Color plus spacing only: type-macro AUROC `0.8448`; removing both: `0.3680`.
- Exact spacing value `1.2`: 67.0% of pooled reference samples versus 5.2% of
  generated samples.
- Unique-color count versus color-sample count: Spearman `0.648` in the
  reference cohort and `0.795` in the generated cohort.

Every retained generation prompt requests literal CSS and a plausible modern
human-authored style. Sixteen reference sources use 163 assets frozen at 2026
capture time. These protocol facts are first-order confounds, not hidden
implementation details.

## Quick Start

Run from the repository root (the staged package's `code/` directory). Keep
recomputed reports outside the checkout so the audited upload tree remains
unchanged:

```bash
mkdir -p ../webhumanbench-recomputed
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src scripts tests

python scripts/audit_public_release.py \
  --source-manifest benchmark/webhumanbench_v1/source_manifest.json \
  --capture-ledger benchmark/webhumanbench_v1/capture_ledger.json \
  --benchmark-manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_release_candidate_audit_recomputed.json
```

The command verifies the included ledgers and feature-bearing manifest without
claiming to recompute hashes of raw files that this release does not distribute.
See [`RELEASE.md`](RELEASE.md) and [`docs/REPRODUCING.md`](docs/REPRODUCING.md)
for baseline, sensitivity, staging, and restricted capture-rebuild commands.

## Evidence

- [`DATA_CARD.md`](DATA_CARD.md): composition, provenance, intended use, and
  limitations.
- [`docs/WEBHUMANBENCH_V1_RELEASE.md`](docs/WEBHUMANBENCH_V1_RELEASE.md): exact
  candidate facts and diagnostics.
- [`docs/FORMAL_BENCHMARK_GATE.md`](docs/FORMAL_BENCHMARK_GATE.md): frozen
  `pilot_only` coverage gate.
- `results/webhumanbench_v1_sensitivity_diagnostics.json`: complete resplit,
  leave-one-reference-out, score-aggregation, challenge-composition, feature,
  viewport, measurement, and exclusion diagnostics.
- `results/webhumanbench_v1_protocol_audit.json`: prompt and capture
  intervention audit.
- `results/webhumanbench_v1_prompt_factorial.json` and
  `results/webhumanbench_v1_prompt_factorial_browser_patch_audit.json`: frozen
  2x2 prompt diagnostic and two-patch Chromium replay.
- `results/webhumanbench_v1_visual_embedding_baselines.json`: frozen UIClip
  image-only control; it is not a preference or authorship evaluation.

## Layout

```text
code/
|-- benchmark/webhumanbench_v1/  # Candidate manifests, ledgers, and feature records
|-- docs/                        # Contracts and reproduction guide
|-- results/                     # Frozen audit, baseline, profile, and diagnostics
|-- scripts/                     # Build, capture, audit, analysis, and staging commands
|-- src/webmark/                 # Feature, baseline, provenance, and audit implementation
`-- tests/                       # Offline regression tests
```

Historical controller and CSS-alignment files remain construction history and
are not evidence for the v1 benchmark paper.

## Rights

Implementation code is Apache-2.0. Benchmark metadata and derived features are
CC-BY-4.0. The anonymous analysis release excludes source captures, vendor
assets, images, fonts, generated HTML, raw provider responses, and screenshots;
their hashes remain in the provenance ledgers. Review
[`THIRD_PARTY_ARTIFACT_NOTICES.md`](THIRD_PARTY_ARTIFACT_NOTICES.md) before
requesting or redistributing any restricted material.

```
