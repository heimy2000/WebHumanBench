# WebHumanBench v1.0.0 Release Candidate

## Identity and Status

- Benchmark: `WebHumanBench`
- Candidate version: `1.0.0`
- Internal artifact revision:
  `39d65e8b308b591075fff1e2ca4a620d4f665d212d9de7bd549721b7bfcdb6c8`
- Source manifest digest:
  `7067abb01232561aaa4b2d699a94a4ea7715cd7655e49927031589ef635b2f98`
- Scoring viewport: `390x844`
- Paired reference viewport: `430x932`
- Browser: Chromium `150.0.7871.115`
- Status: locally staged anonymous analysis-release candidate; no external
  repository URL is asserted

The candidate is a versioned corpus contract for reference-relative browser-feature
fit. It is not an authorship detector, preference dataset, design-quality
standard, accessibility benchmark, or mature leaderboard.

## Corpus

| Page type | Reference sources | Train | Dev | Test | Generated retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| SaaS landing | 6 | 3 | 1 | 2 | 99 |
| Docs homepage | 8 | 4 | 1 | 3 | 100 |
| Product showcase | 4 | 2 | 1 | 1 | 95 |
| Developer tool | 4 | 2 | 1 | 1 | 100 |
| Dashboard shell | 4 | 2 | 1 | 1 | 100 |
| Portfolio showcase | 6 | 3 | 1 | 2 | 96 |
| Total | 32 | 16 | 6 | 10 | 590 |

The generated source run contains 600 completed responses. Ten pages are
excluded before reference scoring because their archived HTML requests a remote
placeholder image. `ai_source_run_ledger.json` and
`ai_exclusion_ledger.json` close the run as `590 + 10` with no score-based
selection.

## Protocol Audit

- Commit timestamps: 1 source in 2019, 3 in 2020, 4 in 2021, 24 in 2022.
- Capture-time vendoring: 163 assets across 16 sources.
- Snapshot interventions: 17 modified entrypoints and 3 removed known
  nonvisual external scripts.
- Prompt audit: 590/590 retained prompts contain the four declared production,
  closure, literal-CSS, and human-authored-style conditions.
- Normalized generated templates: 6; unique system prompts: 1.

The fixed historical-commit rule applies to repository provenance. It does not
date every remotely hosted asset frozen in 2026. The generated cohort is extractor-aware
and prompt-conditioned, not a natural sample of model output.

## Frozen Baselines

| Baseline | Equal-page-type macro AUROC | 95% stratified CI |
| --- | ---: | --- |
| Profile L2+W1 | 0.7910 | [0.6875, 0.8961] |
| Diagonal L2 | 0.8142 | [0.7113, 0.9124] |
| Nearest-train L1 | 0.7945 | [0.6995, 0.8825] |
| Robust MAD L1 | 0.8426 | [0.7620, 0.9233] |
| Typography W1 | 0.3790 | [0.3012, 0.4571] |

All methods fit reference training groups only. The primary endpoint averages
within-type AUROCs with equal type weight; pooled cross-type scores are retained
only for compatibility. All multi-signal intervals overlap, and every type has
only one development source and one to three reference test sources. Per-type
intervals and a winning-baseline claim are not supported.

## Sensitivity Results

- Profile over 100 alternative source assignments: median `0.6234`, IQR
  `[0.5470, 0.6933]`, range `[0.4171, 0.8541]`.
- Robust MAD frozen result is at the 99th percentile of its alternative-split
  distribution; 25% of alternatives are below `0.5`.
- Profile leave-one-reference-out coverage: mean `0.6582`, 95% stratified
  source-bootstrap CI `[0.5666, 0.7433]`, range `[0, 1]`; dashboard mean
  `0.155` versus frozen dashboard AUROC `1.0`.
- Profile across recorded model strata: `[0.597, 0.899]`.
- Profile aggregation: equal-type `0.7910`, pair-weighted within-type `0.7086`,
  and pooled cross-type compatibility value `0.6934`.
- Equal-model Profile mean: `0.7881`; leave-one-model-out range:
  `[0.7731, 0.8200]`.
- Color plus spacing only: `0.8448 [0.7818, 0.9094]`.
- Without color and spacing: `0.3680 [0.2648, 0.4685]`.
- Pooled exact spacing value `1.2`: `0.670` reference versus `0.052` generated.
- Palette cardinality/sample-count Spearman: `0.648` reference versus `0.795`
  generated.

These results support a narrow protocol-conditioned fit diagnostic and expose
construct weakness. They do not support a broad model of human design.

## Frozen Prompt and Visual Controls

- A 2x2, 30-block prompt factorial with one recorded Qwen/Qwen3-32B source run
  finds neutral `0.6444`, literal-only `0.6167`, style-only `0.6111`, and full
  `0.5722` Profile AUROCs. Literal-CSS paired closer-fit rate is `0.5000`
  (`[0.3667, 0.6333]`); style is `0.4000` (`[0.2333, 0.5667]`). These clauses do
  not show reliable improvement at this scale.
- Replaying the same 120 frozen HTML documents in Chromium `150.0.7871.115` and
  `150.0.7871.129` produces 120/120 pixel-identical PNGs, exact feature arrays,
  and exact downstream analysis objects. This is not a cross-engine claim.
- Image-only UIClip controls use no browser feature vector, prompt, or rating.
  Centroid cosine gives `0.8052 [0.7854, 0.8236]`; nearest-reference cosine
  gives `0.7342 [0.7120, 0.7567]`. Product-showcase performance is `0.1158` and
  `0.0632`, respectively, so the encoder does not provide a stable universal
  construct.

## Readiness Status

`audit_benchmark_readiness.py` returns `pilot_only`. The planned target is
120/40/40 reference train/dev/test groups and 100 generated groups per page
type. This gate reflects the small reference cohort, split instability, and
unresolved source, prompt, asset, provider, encoder, and content dependence.

## Anonymous Release Scope

The `1.0.0-rc5` analysis-only package contains the implementation, tests,
manifest and ledger objects, derived feature arrays, frozen results, hashes,
and validation commands. It excludes source captures, snapshot overlays,
vendored assets, generated HTML, raw provider responses, and screenshots.
Consequently, it can reproduce the manifest-integrity check and train-only
feature baselines, but not a byte-level raw-artifact audit or browser recapture
without separately authorized materials. This boundary is intentional and does
not grant redistribution rights for excluded artifacts.

## Promotion Checklist

Before describing v1.0.0 as released:

1. stage the exact candidate with both AI ledgers and both diagnostic artifacts;
2. run package preflight and artifact audit on the staged tree;
3. verify third-party notices and redistribution scope;
4. deposit the staged tree at a real anonymous URL; and
5. update `CITATION.cff` with the actual release date and repository identifier.
