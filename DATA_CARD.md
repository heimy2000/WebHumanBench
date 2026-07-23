# WebHumanBench v1.0.0 Release-Candidate Data Card

## Identity

- Dataset: `WebHumanBench`
- Candidate version: `1.0.0`
- Internal artifact revision:
  `39d65e8b308b591075fff1e2ca4a620d4f665d212d9de7bd549721b7bfcdb6c8`
- Status: locally frozen and audited; `1.0.0-rc5` is staged as an anonymous
  analysis-only release candidate, but no external repository URL is asserted
- Intended task: reference-relative browser-feature fit diagnostics
- Explicitly out of scope: authorship, preference, aesthetic quality,
  usability, accessibility, and universal design norms

Legacy manifest values `human` and `ai` mean historical-reference and generated
challenge provenance. They are not perceptual or verified-authorship labels.

## Composition

| Component | Count |
| --- | ---: |
| Historical-reference source groups | 32 |
| Reference captures | 64 |
| Reference train / dev / test groups | 16 / 6 / 10 |
| Completed generation responses | 600 |
| Generated retained / excluded groups | 590 / 10 |
| Total scoring records | 622 |
| Recorded model identifiers | 7 |

Reference counts by page type are 6 SaaS landings, 8 documentation homepages,
4 product showcases, 4 developer tools, 4 dashboard shells, and 6 portfolios.
Retained generated counts are 99, 100, 95, 100, 100, and 96, respectively.

## Historical-Reference Provenance

Every source has a public repository, pinned commit, repository-local HTML
entrypoint, permissive license evidence, source-project identity, and local
render. The frozen commits range from 2019 to 2022: 1 source is dated 2019, 3
are dated 2020, 4 are dated 2021, and 24 are dated 2022. These records provide
revision traceability, not proof of individual authorship or absence of software
assistance.

Declared licenses are 24 MIT, 5 Apache-2.0, 2 BSD-3-Clause, and 1 CC0-1.0.
One otherwise screened source was rejected before packaging because a
conservative preflight detected a credential-like token pattern in unredacted
upstream HTML. It was not modified or redistributed.

## Capture Boundary

Reference captures use Chromium `150.0.7871.115`, `en-US`, UTC, light color
scheme, reduced motion, device scale factor 1, and mobile viewports `390x844`
and `430x932`. The first is the scoring viewport. Loopback serving and a
local-only network rule make capture deterministic from the package.

Sixteen of 32 sources require 163 remote visual dependencies to be frozen at
2026 capture time, and three known nonvisual external scripts are removed. In
total, 17 snapshot entrypoints differ from the pinned repository entrypoint due
to URL rewriting or script removal. Vendoring makes the capture reproducible;
it does not establish that each remote asset matches bytes served when the
pinned commit was authored.

## Generated Challenge Boundary

The source run completed 600 responses across seven provider-recorded model
identifiers. The benchmark retains the initial generated HTML, not later CSS
controller output. Ten pages are excluded before scoring because their archived
HTML requests a remote placeholder image. The independent source-run and
exclusion ledgers verify `600 = 590 + 10`; eight exclusions belong to the
recorded Hunyuan identifier and two to Qwen.

All 590 retained prompts use one system instruction and six page-type
templates. Every prompt requests:

- self-contained, production-like HTML/CSS;
- no external assets or network calls;
- literal CSS values readable by the extractor; and
- a plausible modern human-authored style.

The generated cohort is therefore prompt- and pipeline-conditioned, not an
uncontrolled sample of model output. Model identifiers are provider-recorded
strings and do not independently verify model weights or implementations.

## Split and Measurement

The deterministic split assigns whole source groups within page type. AI rows
are test only. `group_id` is the reporting unit and `leakage_group_id` prevents
known source, template, or prompt lineages from crossing splits.

`computed-style-v3` records font-size samples, line-height/font-size ratios,
an 8px left-coordinate phase proxy, computed foreground/background colors, and
HSV saturation. These proxies omit semantics, components, interaction,
responsive behavior beyond two widths, usability, and accessibility. A
zero-variance train-only family is marked inactive instead of receiving an
arbitrary scale.

## Frozen Results and Stress Tests

| Baseline | Equal-page-type macro AUROC | 95% stratified CI |
| --- | ---: | --- |
| Profile L2+W1 | 0.7910 | [0.6875, 0.8961] |
| Diagonal L2 | 0.8142 | [0.7113, 0.9124] |
| Nearest-train L1 | 0.7945 | [0.6995, 0.8825] |
| Robust MAD L1 | 0.8426 | [0.7620, 0.9233] |
| Typography W1 | 0.3790 | [0.3012, 0.4571] |

These are corpus-conditional ordering diagnostics. The decisive qualifications
are:

- Profile type-macro AUROC over 100 alternative source splits has median
  `0.6234`, IQR `[0.5470, 0.6933]`, and range `[0.4171, 0.8541]`.
- Leave-one-reference-out Profile coverage over all 32 sources has mean
  `0.6582`, 95% stratified source-bootstrap CI `[0.5666, 0.7433]`, and range
  `[0, 1]`; the dashboard type mean is `0.155`.
- Profile is `0.7086` under within-type pair weighting and `0.6934` under the
  retained pooled cross-type compatibility endpoint; raw type-specific scores
  are not assumed cross-type comparable.
- Equal-model Profile mean is `0.7881`; leave-one-model-out spans
  `[0.7731, 0.8200]`.
- Color plus spacing only obtains `0.8448`; removing both obtains `0.3680`.
- Exact spacing value `1.2` is 67.0% of pooled reference spacing samples and
  5.2% of generated samples.
- Unique-color count correlates with color-sample count (`rho=0.648` reference,
  `rho=0.795` generated).
- Profile type-macro AUROC ranges from `0.597` to `0.899` across recorded model
  strata.

The formal readiness result is `pilot_only`. The registered mature profile
requires 1,200 historical-reference groups and at least 600 generated groups.

## Recommended Use

- Recompute artifact, split, exclusion, and protocol audits.
- Evaluate reference-relative measurements under the declared revision.
- Develop features and controls that reduce source, prompt, asset, and capture
  dependence.
- Report benchmark version, viewport, split, group counts, and `pilot_only`
  status with every result.

## Prohibited Interpretation

- Do not infer whether a person or model authored an arbitrary page.
- Do not rank people, cultures, models, or products by aesthetic quality.
- Do not claim preference, usability, accessibility, WCAG, or security results.
- Do not generalize to commercial, dynamic, multilingual, desktop, or
  cross-browser webpages.
- Do not call v1 a mature leaderboard or a causally controlled benchmark.

## Distribution and Rights

Code is Apache-2.0; metadata and derived arrays are CC-BY-4.0. The anonymous
analysis-release profile includes code, manifests, derived arrays, hashes,
results, and validation only. It excludes source captures, vendor assets,
generated HTML, raw responses, and screenshots, which retain upstream or
provider terms. The staged candidate also excludes caches, AppleDouble metadata,
credentials, symlinks, and unreferenced construction files. Review
`THIRD_PARTY_ARTIFACT_NOTICES.md` before requesting or redistributing any raw
artifact.
