import pytest

from ml_core.embeddings import BagOfWordsEmbedding, TfidfEmbedding

CORPUS = [
    "the human wrote a careful sentence",
    "the model generated fluent text",
    "humans and models both produce text",
]


def test_bow_fit_transform_shape_and_vocab():
    bow = BagOfWordsEmbedding()
    matrix = bow.fit_transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)
    assert matrix.shape[1] == bow.vocabulary_size
    # counts are non-negative integers
    assert matrix.min() >= 0


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        BagOfWordsEmbedding().transform(CORPUS)


def test_tfidf_values_are_weighted():
    tfidf = TfidfEmbedding()
    matrix = tfidf.fit_transform(CORPUS)
    # tf-idf weights are not raw counts; max weight is bounded above by 1 here
    assert matrix.max() <= 1.0 + 1e-9
    assert matrix.shape[0] == len(CORPUS)


def test_fit_then_transform_matches_fit_transform():
    a = TfidfEmbedding()
    m1 = a.fit_transform(CORPUS)
    b = TfidfEmbedding()
    m2 = b.fit(CORPUS).transform(CORPUS)
    assert (m1 != m2).nnz == 0


def test_save_load_roundtrip(tmp_path):
    tfidf = TfidfEmbedding()
    original = tfidf.fit_transform(CORPUS)
    path = tmp_path / "tfidf.joblib"
    tfidf.save(path)

    reloaded = TfidfEmbedding.load(path)
    assert (reloaded.transform(CORPUS) != original).nnz == 0


def test_load_rejects_wrong_type(tmp_path):
    path = tmp_path / "bow.joblib"
    BagOfWordsEmbedding().fit(CORPUS).save(path)
    with pytest.raises(ValueError):
        TfidfEmbedding.load(path)
