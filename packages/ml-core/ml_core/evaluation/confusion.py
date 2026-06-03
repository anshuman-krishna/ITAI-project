from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from sklearn.metrics import confusion_matrix as _confusion_matrix


@dataclass(frozen=True)
class ConfusionMatrixResult:
    labels: list[str]
    matrix: list[list[int]]  # rows = true class, columns = predicted class
    false_positives: dict[str, int]
    false_negatives: dict[str, int]
    support: dict[str, int]  # true-class counts

    def to_dict(self) -> dict:
        return asdict(self)


def confusion_matrix_result(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[object] | None = None,
) -> ConfusionMatrixResult:
    # labels are coerced to strings so the result is json-friendly and order is stable.
    if labels is None:
        labels = sorted({str(x) for x in list(y_true) + list(y_pred)})
    else:
        labels = [str(x) for x in labels]

    y_true_s = [str(x) for x in y_true]
    y_pred_s = [str(x) for x in y_pred]
    matrix = _confusion_matrix(y_true_s, y_pred_s, labels=labels)

    false_positives: dict[str, int] = {}
    false_negatives: dict[str, int] = {}
    support: dict[str, int] = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        row = int(matrix[i, :].sum())  # everything truly this class
        col = int(matrix[:, i].sum())  # everything predicted this class
        false_negatives[label] = row - tp
        false_positives[label] = col - tp
        support[label] = row

    return ConfusionMatrixResult(
        labels=list(labels),
        matrix=matrix.tolist(),
        false_positives=false_positives,
        false_negatives=false_negatives,
        support=support,
    )
