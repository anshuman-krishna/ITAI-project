# AI-text detection benchmark (primary objective)

Dataset: `train_essays` — 1378 samples. Evaluation: stratified 5-fold cross-validation, identical splits and seed across all cells.

## Methodology

Every embedding is paired with every classifier and evaluated under identical data,
splits, and random seed, so the only variables that change between cells are the
embedding and the classifier. The embedding is refit on each training fold and only
transforms the held-out fold, so no test information leaks into the features. Metrics
are averaged across folds; both weighted and macro averages are reported.

- Preprocessing: lowercase, remove_punctuation, normalize_whitespace, remove_stopwords
- Embeddings: bow, tfidf
- Classifiers: linear_svm, logistic_regression, naive_bayes

## Results

Ranked by f1_weighted.

| embedding | classifier | accuracy | f1 (weighted) | f1 (macro) | precision (macro) | recall (macro) | fit ms | predict ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bow | logistic_regression | 0.998 | 0.997 | 0.700 | 0.699 | 0.700 | 12.91 | 0.093 |
| bow | naive_bayes | 0.998 | 0.997 | 0.700 | 0.699 | 0.700 | 0.83 | 0.155 |
| bow | linear_svm | 0.998 | 0.997 | 0.700 | 0.699 | 0.700 | 6.41 | 0.090 |
| tfidf | logistic_regression | 0.998 | 0.997 | 0.700 | 0.699 | 0.700 | 7.55 | 0.088 |
| tfidf | naive_bayes | 0.998 | 0.997 | 0.700 | 0.699 | 0.700 | 0.80 | 0.138 |
| tfidf | linear_svm | 0.998 | 0.997 | 0.700 | 0.699 | 0.700 | 2.74 | 0.086 |

## Observations

- Key finding: the dataset is unsuitable for meaningful AI-vs-human benchmarking. The official train_essays.csv contains only 3 AI-generated essays out of 1,378 (class-imbalance ratio 458:1), so every classifier collapses to predicting the majority class 'human' for all samples.
- Accuracy and weighted-F1 near 0.998 are the majority-class baseline, not detection skill; the AI class is never predicted, so its recall is 0 and macro-F1 is capped near 0.70.
- A genuine detection benchmark requires augmenting this corpus with AI-generated essays for the same prompts (see Limitations and Future Work).
- Highest f1_weighted: bow + logistic_regression (0.997).
- Highest macro-F1: bow + logistic_regression (0.699); macro-F1 is the fairer measure when classes are imbalanced.
- Fastest to train: tfidf + naive_bayes (0.80 ms/fold).
- Fastest to predict: tfidf + linear_svm (0.086 ms/fold).
