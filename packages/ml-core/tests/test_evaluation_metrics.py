from ml_core.evaluation import (
    compute_metrics,
    confusion_matrix_result,
    summarize_metrics,
)


def test_compute_metrics_has_macro_and_weighted():
    metrics = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert metrics["accuracy"] == 1.0
    for key in ("precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted"):
        assert key in metrics


def test_summarize_metrics_mean_and_std():
    folds = [{"accuracy": 0.8}, {"accuracy": 1.0}]
    summary = summarize_metrics(folds)
    assert summary["accuracy"].mean == 0.9
    assert summary["accuracy"].std > 0
    assert summary["accuracy"].values == [0.8, 1.0]


def test_summarize_empty_is_empty():
    assert summarize_metrics([]) == {}


def test_confusion_counts_fp_fn():
    # true: 0,0,1,1  pred: 0,1,1,1 -> for class "0": fn=1 (one 0 predicted 1), fp=0
    cm = confusion_matrix_result([0, 0, 1, 1], [0, 1, 1, 1])
    assert cm.labels == ["0", "1"]
    assert cm.support == {"0": 2, "1": 2}
    assert cm.false_negatives["0"] == 1
    assert cm.false_positives["1"] == 1
