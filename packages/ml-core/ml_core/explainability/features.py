from __future__ import annotations

from dataclasses import asdict, dataclass

from ml_core.inference import DetectorPipeline


class ExplainabilityNotSupported(RuntimeError):
    # raised when a model has no linear decision surface to read weights from, or an
    # embedding without named features.
    pass


@dataclass(frozen=True)
class FeatureWeight:
    token: str
    weight: float


@dataclass(frozen=True)
class FeatureImportance:
    """the tokens that most push a binary linear model toward each class.

    weights come straight from the fitted model's coefficients, mapped back to the
    embedding's vocabulary. positive weights push toward `positive_class`, negative toward
    `negative_class` — the standard reading of a binary linear decision boundary.
    """

    model: str
    positive_class: str
    negative_class: str
    top_positive: list[FeatureWeight]
    top_negative: list[FeatureWeight]

    def to_dict(self) -> dict:
        return asdict(self)


def explain_pipeline(pipeline: DetectorPipeline, *, top_n: int = 20) -> FeatureImportance:
    """top weighted features for a fitted linear pipeline.

    practical, dependency-free explainability: no SHAP, just the model's own coefficients.
    only defined for binary linear classifiers over a named-feature embedding, which covers
    the platform's logistic-regression and linear-svm baselines over bag-of-words / tf-idf.
    """
    coef = pipeline.classifier.linear_coefficients()
    if coef is None:
        raise ExplainabilityNotSupported(
            f"{pipeline.classifier_name} has no linear coefficients to explain"
        )

    classes = pipeline.classifier.classes_
    if len(classes) != 2 or coef.shape[0] != 1:
        raise ExplainabilityNotSupported("feature importance is defined for binary linear models")

    try:
        names = pipeline.embedding.feature_names()
    except NotImplementedError as exc:
        raise ExplainabilityNotSupported(str(exc)) from exc

    weights = coef[0]
    paired = sorted(zip(names, weights), key=lambda x: x[1])
    top_negative = [FeatureWeight(token, float(w)) for token, w in paired[:top_n]]
    top_positive = [FeatureWeight(token, float(w)) for token, w in reversed(paired[-top_n:])]

    return FeatureImportance(
        model=pipeline.model_name,
        positive_class=pipeline.display_label(classes[1]),
        negative_class=pipeline.display_label(classes[0]),
        top_positive=top_positive,
        top_negative=top_negative,
    )
