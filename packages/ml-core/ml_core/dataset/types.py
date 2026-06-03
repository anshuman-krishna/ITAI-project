from __future__ import annotations

from dataclasses import asdict, dataclass

from ml_core.analytics import ClassDistribution, LengthStats, VocabularyStats


@dataclass(frozen=True)
class DatasetProfile:
    total_samples: int
    n_columns: int
    char_length: LengthStats
    word_length: LengthStats
    vocabulary: VocabularyStats
    class_distribution: ClassDistribution | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" or "warning"
    code: str
    message: str
    count: int


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool  # false when any error-severity issue is present
    total_samples: int
    issues: list[ValidationIssue]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]
