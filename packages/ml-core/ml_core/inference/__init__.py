"""inference: a trained pipeline that turns raw text into a prediction.

DetectorPipeline bundles preprocessing, a fitted embedding, and a fitted classifier into
one serializable unit, reusing the same components as the experiment runner (no duplicated
ml logic). it can be saved to a directory of artifacts and reloaded so predictions never
retrain. PredictionResult carries the label, confidence, full probabilities, and metadata.
"""

from ml_core.inference.pipeline import DetectorPipeline, PredictionResult

__all__ = ["DetectorPipeline", "PredictionResult"]
