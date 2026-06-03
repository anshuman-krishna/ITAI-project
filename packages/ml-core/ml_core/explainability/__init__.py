"""explainability: why the model decided what it did.

practical and deliberately small for an mvp — top weighted features (model coefficients
mapped back to tokens) for binary linear classifiers, no SHAP. predictions stay
inspectable: a reader can see which tokens push a text toward each class.
"""

from ml_core.explainability.features import (
    ExplainabilityNotSupported,
    FeatureImportance,
    FeatureWeight,
    explain_pipeline,
)

__all__ = [
    "ExplainabilityNotSupported",
    "FeatureImportance",
    "FeatureWeight",
    "explain_pipeline",
]
