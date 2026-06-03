# System-validation benchmark — prompt classification (auxiliary)

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
| bow | logistic_regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11.55 | 0.092 |
| bow | naive_bayes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.89 | 0.150 |
| bow | linear_svm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.88 | 0.096 |
| tfidf | naive_bayes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 | 0.161 |
| tfidf | logistic_regression | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 9.93 | 0.088 |
| tfidf | linear_svm | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 3.01 | 0.083 |

## Observations

- This is NOT the AI-detection task. It is a system-validation benchmark: a balanced, real-text binary classification (essay prompt 0 vs 1, 708 vs 670 samples) drawn from the same corpus, used to confirm the benchmarking pipeline produces meaningful, differentiated results on real data.
- Because the classes are balanced, weighted and macro metrics agree and the confusion matrices are informative — in contrast to the degenerate detection benchmark.
- Highest f1_weighted: bow + logistic_regression (1.000).
- Highest macro-F1: bow + logistic_regression (1.000); macro-F1 is the fairer measure when classes are imbalanced.
- Fastest to train: tfidf + naive_bayes (0.78 ms/fold).
- Fastest to predict: tfidf + linear_svm (0.083 ms/fold).
