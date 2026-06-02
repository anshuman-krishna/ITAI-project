# ml-core

The machine learning library for the platform. It's one package with strong internal
boundaries rather than several packages — Phase 1 optimizes for research velocity. A
module gets extracted into its own package only when a real boundary demands it.

```
ml_core/
  preprocessing/   composable text cleaning and tokenization pipelines
  embeddings/      interchangeable representations behind one interface
  training/        classifiers that consume any embedding
  evaluation/      metrics, cross-validation, error analysis
  analytics/       dataset analysis and quality scoring
  visualization/   plots for datasets, embeddings, and results
```

## The embedding contract

Every representation implements `embeddings.Embedding`:

```python
fit(texts) -> Embedding
transform(texts) -> np.ndarray
fit_transform(texts) -> np.ndarray
save(path) -> None
load(path) -> Embedding
```

Classifiers depend on this interface, never on a concrete embedding. That's what makes
comparing bag-of-words, tf-idf, word2vec, doc2vec, and transformer features fair: same
dataset, same split, same evaluation, one variable changed at a time.

## Install

```bash
pip install -e ".[dev]"
```

Dependencies are kept minimal (numpy, pandas, scikit-learn). Heavier ones (gensim, torch,
transformers) are added to a module only when that module is implemented.
