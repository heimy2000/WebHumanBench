"""Reference implementation and release tooling for WebHumanBench v1.

Public API
----------
- :func:`evaluate_reference_fit_baselines` — train-only, type-conditioned
  reference-fit baselines used by the benchmark.
- :func:`audit_benchmark_readiness` and :func:`validate_public_release` —
  fail-closed benchmark and artifact-integrity gates.
- :func:`compute_composite_bias` and :class:`BiasScorer` — legacy reference
  scores retained for compatibility with earlier experimental artifacts.
- :func:`apply_operator` and :func:`beam_search_correct` — legacy CSS
  correction utilities; they are not part of the WebHumanBench v1 task.
- :class:`PageTypeClassifier` — analysis-only 6-class diagnostic wrapper.
- :func:`validate_public_release` — cross-artifact integrity audit required
  before publishing a source manifest, capture ledger, and benchmark manifest.
- :func:`sign_test`, :func:`cluster_bootstrap_ci`, :func:`cohens_d`,
  :func:`holm_bonferroni`, :func:`post_hoc_power` — statistical audit
  primitives for separately justified analyses; they are not evidence for the
  heterogeneous SE-18 bundle.
"""

from .beam_search import (
    BeamSearchConfig,
    beam_search_correct,
)
from .benchmark_baselines import evaluate_reference_fit_baselines
from .benchmark_readiness import audit_benchmark_readiness
from .bias import (
    BiasScorer,
    ReferenceStats,
    compute_composite_bias,
    gaussian_zscore,
    wasserstein1_distance,
)
from .browser_alignment import audit_browser_scored_alignment, evaluate_browser_scored_alignment
from .contrast import (
    contrast_exposure_delta,
    static_low_contrast_exposure,
)
from .features import (
    FEATURE_NAMES,
    PageFeatures,
    extract_page_features,
)
from .human_likeness import (
    SCHEMA as HUMAN_LIKENESS_BENCHMARK_SCHEMA,
)
from .human_likeness import (
    BenchmarkRecord,
    evaluate_human_likeness_benchmark,
    fit_reference,
    human_fit_percentile,
    validate_manifest,
)
from .open_reference import (
    SCHEMA as OPEN_REFERENCE_SCHEMA,
)
from .open_reference import (
    OpenReferenceSource,
    open_reference_summary,
    validate_open_reference_manifest,
)
from .operators import (
    OPERATOR_CATALOG,
    OperatorName,
    apply_operator,
)
from .page_type import (
    PAGE_TYPE_ID_TO_NAME,
    PAGE_TYPES,
    PageTypeClassifier,
)
from .release import (
    CAPTURE_SCHEMA,
    PUBLIC_RELEASE_SCHEMA,
    canonical_json_sha256,
    sha256_file,
    validate_public_release,
)
from .scale_temporal_validation import (
    audit_scale_temporal_panel,
    build_scale_temporal_panel,
    run_scale_temporal_validation,
)
from .scoring import (
    UICLIP_NORMALIZATION,
    AestheticV25Scorer,
    UIClipScorer,
)
from .stats_utils import (
    cluster_bootstrap_ci,
    cohens_d_paired,
    exact_binomial_pvalue,
    holm_bonferroni,
    post_hoc_power_binomial,
    sign_test,
    t_test_ci_paired,
)

__version__ = "1.0.0rc1"

__all__ = [
    # bias
    "BiasScorer",
    "ReferenceStats",
    "compute_composite_bias",
    "gaussian_zscore",
    "wasserstein1_distance",
    # beam search
    "BeamSearchConfig",
    "beam_search_correct",
    # features
    "FEATURE_NAMES",
    "PageFeatures",
    "extract_page_features",
    # contrast guardrail
    "contrast_exposure_delta",
    "static_low_contrast_exposure",
    # operators
    "OPERATOR_CATALOG",
    "OperatorName",
    "apply_operator",
    # page type
    "PAGE_TYPES",
    "PAGE_TYPE_ID_TO_NAME",
    "PageTypeClassifier",
    # human-likeness benchmark
    "HUMAN_LIKENESS_BENCHMARK_SCHEMA",
    "BenchmarkRecord",
    "evaluate_human_likeness_benchmark",
    "fit_reference",
    "human_fit_percentile",
    "validate_manifest",
    "evaluate_reference_fit_baselines",
    "audit_benchmark_readiness",
    "evaluate_browser_scored_alignment",
    "audit_browser_scored_alignment",
    # open mobile reference contract
    "OPEN_REFERENCE_SCHEMA",
    "OpenReferenceSource",
    "open_reference_summary",
    "validate_open_reference_manifest",
    # public-release integrity contract
    "CAPTURE_SCHEMA",
    "PUBLIC_RELEASE_SCHEMA",
    "canonical_json_sha256",
    "sha256_file",
    "validate_public_release",
    # scale and temporal-control validation
    "audit_scale_temporal_panel",
    "build_scale_temporal_panel",
    "run_scale_temporal_validation",
    # scoring
    "UICLIP_NORMALIZATION",
    "AestheticV25Scorer",
    "UIClipScorer",
    # stats
    "cluster_bootstrap_ci",
    "cohens_d_paired",
    "exact_binomial_pvalue",
    "holm_bonferroni",
    "post_hoc_power_binomial",
    "sign_test",
    "t_test_ci_paired",
    # meta
    "__version__",
]
