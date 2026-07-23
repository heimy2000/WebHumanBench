"""Page-type classification utilities for the release diagnostic.

The correction path in :mod:`webmark.beam_search` is page-type agnostic. This
module is used only by the SE-8 diagnostic that tests whether CSS summary
features can recover the page-type labels attached to the generated pages.
It must not be described as part of the released correction controller.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

from .features import PageFeatures

PAGE_TYPES: tuple[str, ...] = (
    "saas_landing",
    "docs_homepage",
    "product_showcase",
    "developer_tool",
    "dashboard_shell",
    "portfolio_showcase",
)
PAGE_TYPE_ID_TO_NAME: dict[int, str] = {i: name for i, name in enumerate(PAGE_TYPES)}

CLASSIFIER_FEATURE_NAMES: tuple[str, ...] = (
    "typography_mean",
    "typography_std",
    "typography_n",
    "spacing_mean",
    "spacing_std",
    "spacing_n",
    "grid_mean",
    "grid_std",
    "grid_n",
    "color_unique",
    "color_n",
    "saturation_mean",
    "saturation_std",
    "saturation_n",
)


def classifier_feature_vector(features: PageFeatures) -> list[float]:
    """Return the fixed 14-value CSS summary used by the SE-8 diagnostic."""

    summary = features.to_summary()
    return [float(summary[name]) for name in CLASSIFIER_FEATURE_NAMES]


def make_page_type_pipeline(*, random_state: int = 42):
    """Build the exact L2 multinomial pipeline used by the SE-8 diagnostic."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Page-type diagnostics require scikit-learn; install webmark[analysis]."
        ) from exc

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class PageTypePrediction:
    """One diagnostic prediction with per-class posterior probabilities."""

    name: str
    confidence: float
    posteriors: dict[str, float]


class PageTypeClassifier:
    """Thin fitted-model wrapper for analysis code, not correction routing."""

    def __init__(self, model_path: pathlib.Path | None = None) -> None:
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[str],
        *,
        random_state: int = 42,
    ) -> PageTypeClassifier:
        self.model = make_page_type_pipeline(random_state=random_state)
        self.model.fit(X, y)
        return self

    def predict(self, features_summary: Sequence[float]) -> PageTypePrediction:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")

        probs = self.model.predict_proba([features_summary])[0]
        classes = [str(label) for label in self.model.classes_]
        top_idx = int(max(range(len(probs)), key=lambda idx: probs[idx]))
        return PageTypePrediction(
            name=classes[top_idx],
            confidence=float(probs[top_idx]),
            posteriors={label: float(prob) for label, prob in zip(classes, probs, strict=False)},
        )

    def save(self, path: pathlib.Path) -> None:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: pathlib.Path) -> PageTypeClassifier:
        import joblib

        self.model = joblib.load(path)
        return self
