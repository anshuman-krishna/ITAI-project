from ml_core.embeddings import BagOfWordsEmbedding
from ml_core.models import MultinomialNaiveBayes
from ml_core.training import cross_validate

from tests.data import LABELS, TEXTS


def _run(seed=42):
    return cross_validate(
        lambda: BagOfWordsEmbedding(),
        lambda: MultinomialNaiveBayes(),
        TEXTS,
        LABELS,
        n_folds=3,
        seed=seed,
    )


def test_cross_validate_shape():
    result = _run()
    assert result.n_folds == 3
    assert len(result.fold_metrics) == 3
    assert "f1_weighted" in result.aggregate
    assert result.confusion.labels == ["0", "1"]
    assert result.mean_fit_seconds >= 0


def test_aggregate_matches_fold_values():
    result = _run()
    summary = result.aggregate["accuracy"]
    assert summary.values == [f["accuracy"] for f in result.fold_metrics]


def test_reproducible_with_same_seed():
    assert _run(7).fold_metrics == _run(7).fold_metrics


def test_separable_data_scores_well():
    # disjoint vocabularies should be easy; sanity check the pipeline actually learns.
    assert _run().aggregate["accuracy"].mean > 0.8
