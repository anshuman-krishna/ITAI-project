# ml-core

The machine learning library for the platform. It's one package with strong internal
boundaries rather than several packages — optimized for research velocity. A module gets
extracted into its own package only when a real boundary demands it.

```
ml_core/
  dataset/         ingestion, canonical schema, validation, profiling
  preprocessing/   composable text cleaning and tokenization pipelines
  embeddings/      interchangeable representations behind one interface
  analytics/       dataset analysis: lengths, vocabulary, class balance
  evaluation/      train/test + stratified splitting, metrics
  experiments/     file-based experiment metadata
  visualization/   plots for datasets, embeddings, and results
  training/        classifiers that consume any embedding (later phase)
```

## Dataset workflow

A raw table becomes a canonical `TextDataset` described by a `TextDatasetSchema`, so column
assumptions never leak across the code. Dataframes stay inside ml-core.

```python
from ml_core.dataset import TextDatasetSchema, load_dataset, profile_dataset, validate_dataset

schema = TextDatasetSchema(text_column="text", label_column="generated")
dataset = load_dataset("train.csv", schema)        # csv now, parquet with the [parquet] extra

profile = profile_dataset(dataset)                 # typed DatasetProfile
report = validate_dataset(dataset, allowed_labels=[0, 1])  # typed ValidationReport
```

`load_records(records, schema)` is the in-memory entry point the API uses (no filesystem).

## Preprocessing workflow

Small composable steps driven by a config, so experiments compare preprocessing choices.
The default path is offline and deterministic.

```python
from ml_core.preprocessing import PreprocessingConfig, compose_pipeline

pipeline = compose_pipeline(PreprocessingConfig(remove_stopwords=True))
cleaned = pipeline.run_many(dataset.texts)
```

Toggles: `lowercase`, `remove_punctuation`, `normalize_whitespace`, `remove_stopwords`,
`lemmatize`. Lemmatization needs the `[lemmatization]` extra (nltk + wordnet data).

## Feature extraction workflow

Bag-of-words and tf-idf implement the `Embedding` contract:

```python
fit(texts) -> Embedding
transform(texts) -> FeatureMatrix     # np.ndarray | scipy sparse matrix
fit_transform(texts) -> FeatureMatrix
save(path) -> None
load(path) -> Embedding
```

```python
from ml_core.embeddings import TfidfEmbedding

tfidf = TfidfEmbedding(max_features=20000, ngram_range=(1, 2))
features = tfidf.fit_transform(cleaned)            # sparse matrix
tfidf.save("artifacts/tfidf.joblib")
```

Classical features are sparse and high-dimensional, so `transform` returns a sparse matrix
rather than forcing dense arrays. Classifiers depend on this interface, never on a concrete
embedding — that's what makes comparing representations fair: same dataset, same split, same
evaluation, one variable at a time.

## Experiment workflow

Classifiers (`models`) implement a `Classifier` interface that mirrors the embeddings:
`fit / predict / predict_proba / save / load`. `predict_proba` raises `ProbabilityNotSupported`
where it doesn't apply (e.g. linear svm). The `ExperimentRunner` is the one place the whole
pipeline is defined — every script, test, and api call goes through it:

```python
from ml_core.experiments import ExperimentConfig, ExperimentRunner
from ml_core.benchmarking import BenchmarkRegistry
from ml_core.preprocessing import PreprocessingConfig

registry = BenchmarkRegistry("reports/benchmarks/baseline.json")
config = ExperimentConfig(
    embedding="tfidf",          # resolved via the embedding registry
    classifier="logistic_regression",
    preprocessing=PreprocessingConfig(remove_stopwords=True),
    n_folds=5,
)
result = ExperimentRunner(registry).run(config, texts, labels)
print(result.metrics["f1_weighted"].mean)
```

Cross-validation refits the embedding on each training fold and only transforms the test
fold, so no test information leaks into the features. Results carry per-fold and aggregate
metrics (macro + weighted), a confusion matrix, and timings.

## Benchmark workflow

The registry stores every run as json and compares them:

```python
from ml_core.benchmarking import rank_results, compare_classifiers, summarize
from ml_core.experiments import run_baseline_suite

run_baseline_suite(texts, labels, registry=registry)   # bow/tfidf x logreg/nb/svm
ranked = registry.rank("f1_weighted")
summary = registry.summarize("f1_weighted")            # best per metric + fastest
```

`scripts/run_benchmark_suite.py` runs the official suite end to end (synthetic data, or a
real csv via `--csv`).

The module dependency graph is acyclic and `evaluation` depends on nothing else in ml-core,
so metrics stay an independent, trustworthy concern.

## Install

```bash
pip install -e ".[dev]"                 # core
pip install -e ".[dev,parquet,lemmatization]"   # with optional features
pytest
```

Core deps (numpy, pandas, scikit-learn, scipy, joblib, matplotlib) stay lean. Heavier ones
(gensim, torch, transformers) are added to a module only when that module is implemented.
