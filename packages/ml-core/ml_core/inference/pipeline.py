from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ml_core.embeddings import make_embedding
from ml_core.embeddings.base import Embedding
from ml_core.models import make_classifier
from ml_core.models.base import Classifier, ProbabilityNotSupported
from ml_core.preprocessing import PreprocessingConfig, compose_pipeline


@dataclass(frozen=True)
class PredictionResult:
    """one prediction with everything a caller needs to trust it.

    label is the (optionally human-readable) predicted class; confidence is the probability
    of that class when the model exposes probabilities, otherwise None. probabilities holds
    the full distribution; metadata records which model produced the result.
    """

    label: str
    confidence: float | None
    probabilities: dict[str, float] | None
    model: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DetectorPipeline:
    """a trained text classifier as one reusable, serializable unit.

    bundles the preprocessing config, the fitted embedding, and the fitted classifier so a
    single object turns raw text into a prediction. it reuses the same preprocessing,
    embedding, and model components as the experiment runner — there is no second copy of
    the ml logic here, only the inference-time composition of it. train once, save, then
    load and predict without retraining.
    """

    def __init__(
        self,
        *,
        embedding: str,
        classifier: str,
        preprocessing: PreprocessingConfig | None = None,
        embedding_params: Mapping[str, object] | None = None,
        classifier_params: Mapping[str, object] | None = None,
        label_names: Mapping[str, str] | None = None,
    ) -> None:
        self.embedding_name = embedding
        self.classifier_name = classifier
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.embedding_params = dict(embedding_params or {})
        self.classifier_params = dict(classifier_params or {})
        # maps raw label -> display name, e.g. {"0": "human", "1": "ai"}.
        self.label_names = dict(label_names or {})
        self._pipeline = compose_pipeline(self.preprocessing)
        self._embedding: Embedding | None = None
        self._classifier: Classifier | None = None
        self._trained_at: str | None = None
        self._n_train: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._classifier is not None and self._embedding is not None

    @property
    def model_name(self) -> str:
        return f"{self.embedding_name}+{self.classifier_name}"

    @property
    def embedding(self) -> Embedding:
        self._require_fitted()
        return self._embedding

    @property
    def classifier(self) -> Classifier:
        self._require_fitted()
        return self._classifier

    def display_label(self, label: object) -> str:
        return self._display(str(label))

    def describe(self) -> dict:
        # public, json-friendly description of the trained model.
        self._require_fitted()
        return self._metadata()

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("pipeline is not fitted; call fit or load first")

    def _display(self, label: str) -> str:
        return self.label_names.get(label, label)

    def fit(self, texts: Sequence[str], labels: Sequence[object]) -> "DetectorPipeline":
        cleaned = self._pipeline.run_many(texts)
        labels = [str(label) for label in labels]

        self._embedding = make_embedding(self.embedding_name, **self.embedding_params)
        features = self._embedding.fit_transform(cleaned)

        self._classifier = make_classifier(self.classifier_name, **self.classifier_params)
        self._classifier.fit(features, labels)

        self._trained_at = datetime.now(timezone.utc).isoformat()
        self._n_train = len(labels)
        return self

    def predict(self, text: str) -> PredictionResult:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Sequence[str]) -> list[PredictionResult]:
        self._require_fitted()
        features = self._embedding.transform(self._pipeline.run_many(texts))
        predicted = [str(p) for p in self._classifier.predict(features)]

        proba_rows = self._proba_rows(features)
        meta = self._metadata()
        results = []
        for i, label in enumerate(predicted):
            probabilities = proba_rows[i] if proba_rows is not None else None
            confidence = probabilities[self._display(label)] if probabilities is not None else None
            results.append(
                PredictionResult(
                    label=self._display(label),
                    confidence=confidence,
                    probabilities=probabilities,
                    model=self.model_name,
                    metadata=meta,
                )
            )
        return results

    def _proba_rows(self, features) -> list[dict[str, float]] | None:
        # full probability distribution per row, keyed by display label, or None when the
        # classifier (e.g. a plain linear svm) does not expose calibrated probabilities.
        if not self._classifier.supports_proba:
            return None
        try:
            matrix = self._classifier.predict_proba(features)
        except ProbabilityNotSupported:
            return None
        classes = [self._display(str(c)) for c in self._classifier.classes_]
        return [{cls: float(row[j]) for j, cls in enumerate(classes)} for row in matrix]

    def _metadata(self) -> dict:
        return {
            "embedding": self.embedding_name,
            "classifier": self.classifier_name,
            "preprocessing": asdict(self.preprocessing),
            "labelNames": self.label_names,
            "trainedAt": self._trained_at,
            "nTrainSamples": self._n_train,
        }

    # --- persistence: a small directory of artifacts, no database required. ---

    _META_FILE = "pipeline.json"
    _EMBEDDING_FILE = "embedding.joblib"
    _CLASSIFIER_FILE = "classifier.joblib"

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._embedding.save(path / self._EMBEDDING_FILE)
        self._classifier.save(path / self._CLASSIFIER_FILE)
        (path / self._META_FILE).write_text(
            json.dumps(
                {
                    "embedding": self.embedding_name,
                    "classifier": self.classifier_name,
                    "preprocessing": asdict(self.preprocessing),
                    "embedding_params": self.embedding_params,
                    "classifier_params": self.classifier_params,
                    "label_names": self.label_names,
                    "trained_at": self._trained_at,
                    "n_train_samples": self._n_train,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DetectorPipeline":
        path = Path(path)
        meta = json.loads((path / cls._META_FILE).read_text())
        pipeline = cls(
            embedding=meta["embedding"],
            classifier=meta["classifier"],
            preprocessing=PreprocessingConfig(**meta["preprocessing"]),
            embedding_params=meta.get("embedding_params"),
            classifier_params=meta.get("classifier_params"),
            label_names=meta.get("label_names"),
        )
        # reconstruct concrete classes via the registries, then load fitted state into them.
        embedding_cls = type(make_embedding(meta["embedding"]))
        classifier_cls = type(make_classifier(meta["classifier"]))
        pipeline._embedding = embedding_cls.load(path / cls._EMBEDDING_FILE)
        pipeline._classifier = classifier_cls.load(path / cls._CLASSIFIER_FILE)
        pipeline._trained_at = meta.get("trained_at")
        pipeline._n_train = meta.get("n_train_samples", 0)
        return pipeline
