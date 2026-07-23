# Fixed-Commit Reference Design Profile

This report describes the v1.0.0 release-candidate fixed-commit historical open-source reference cohort at the two declared mobile viewports.
It is not a human-authorship detector, visual-preference study, or universal design-quality target.

- Source groups: 32
- Mobile captures: 64
- Paired 390x844 / 430x932 source groups: 32
- Overall medians first average each source group across its two declared mobile captures.

## Overall Mobile Profile

| Metric | Source-level median (IQR) |
| --- | --- |
| Text font-size p50 (px) | 16.000 (IQR 0.250) |
| Text font-size IQR (px) | 2.450 (IQR 6.000) |
| Type hierarchy p90 / p50 | 1.500 (IQR 0.701) |
| Line-height ratio p50 | 1.200 (IQR 0.300) |
| Line-height ratio IQR | 0.000 (IQR 0.053) |
| 8px phase-alignment proxy | 0.730 (IQR 0.499) |
| Unique computed colors | 4.500 (IQR 6.250) |
| Top-5 color share | 1.000 (IQR 0.042) |
| Neutral color share (S <= 0.10) | 0.765 (IQR 0.394) |
| Saturation p50 | 0.000 (IQR 0.036) |

## By Page Type

| Page type | Sources | Captures | Type hierarchy | Neutral share |
| --- | ---: | ---: | --- | --- |
| dashboard_shell | 4 | 8 | 1.192 (IQR 0.498) | 0.776 (IQR 0.432) |
| developer_tool | 4 | 8 | 1.100 (IQR 0.303) | 0.754 (IQR 0.275) |
| docs_homepage | 8 | 16 | 1.619 (IQR 0.639) | 0.866 (IQR 0.219) |
| portfolio_showcase | 6 | 12 | 1.635 (IQR 0.204) | 0.702 (IQR 0.538) |
| product_showcase | 4 | 8 | 1.500 (IQR 0.209) | 0.496 (IQR 0.221) |
| saas_landing | 6 | 12 | 1.440 (IQR 0.803) | 0.813 (IQR 0.291) |

## Interpretation Boundary

The reported values are descriptive source-level summaries. Typography and color measurements are browser-computed samples; the 8px value is a left-position phase proxy; and paired viewport differences describe responsive change rather than an aesthetic or accessibility score. The historical cutoff and source-project evidence reduce provenance ambiguity but do not establish individual human authorship or exclude all forms of automated assistance.
