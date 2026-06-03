from __future__ import annotations

from dataclasses import dataclass, field

from ml_core.preprocessing import PreprocessingConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """everything needed to run one experiment, by name.

    embeddings and classifiers are named (resolved through their registries) so the same
    config can be built from an api request, a script, or a test without importing
    concrete classes.
    """

    embedding: str
    classifier: str
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding_params: dict = field(default_factory=dict)
    classifier_params: dict = field(default_factory=dict)
    n_folds: int = 5
    seed: int = 42
    dataset_version: str | None = None
    notes: str = ""
