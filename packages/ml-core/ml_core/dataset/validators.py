from __future__ import annotations

from collections.abc import Collection

from ml_core.analytics import class_distribution
from ml_core.dataset.dataset import TextDataset
from ml_core.dataset.types import ValidationIssue, ValidationReport

# above this largest/smallest class ratio, imbalance is flagged as a warning.
DEFAULT_IMBALANCE_THRESHOLD = 3.0


def validate_dataset(
    dataset: TextDataset,
    *,
    allowed_labels: Collection[object] | None = None,
    imbalance_threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
) -> ValidationReport:
    frame = dataset.frame
    schema = dataset.schema
    issues: list[ValidationIssue] = []

    # empty or whitespace-only texts cannot be learned from.
    texts = dataset.texts
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        issues.append(
            ValidationIssue("error", "empty_text", "rows with empty text", empty)
        )

    # duplicate rows inflate counts and leak across train/test splits.
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        issues.append(
            ValidationIssue("warning", "duplicate_rows", "duplicate rows", duplicates)
        )

    if schema.has_labels:
        labels = frame[schema.label_column]
        missing_labels = int(labels.isna().sum())
        if missing_labels:
            issues.append(
                ValidationIssue(
                    "error", "missing_label", "rows with missing label", missing_labels
                )
            )

        if allowed_labels is not None:
            allowed = {str(x) for x in allowed_labels}
            present = labels.dropna().astype(str)
            invalid = int((~present.isin(allowed)).sum())
            if invalid:
                issues.append(
                    ValidationIssue(
                        "error", "invalid_label", "labels outside allowed set", invalid
                    )
                )

        dist = class_distribution([x for x in dataset.labels if x is not None])
        if dist.n_classes < 2:
            issues.append(
                ValidationIssue(
                    "warning", "single_class", "fewer than two classes present", dist.n_classes
                )
            )
        elif dist.imbalance_ratio > imbalance_threshold:
            issues.append(
                ValidationIssue(
                    "warning",
                    "class_imbalance",
                    f"class imbalance ratio {dist.imbalance_ratio:.1f} exceeds {imbalance_threshold}",
                    dist.n_classes,
                )
            )

    is_valid = not any(i.severity == "error" for i in issues)
    return ValidationReport(is_valid=is_valid, total_samples=len(dataset), issues=issues)
