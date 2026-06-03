"""ml-core: the machine learning library for the platform.

modules are organized by responsibility and kept loosely coupled so embeddings,
classifiers, and evaluation can be combined and benchmarked under identical conditions:

    dataset         ingestion, canonical schema, validation, profiling
    preprocessing   configurable text cleaning and tokenization pipelines
    embeddings      interchangeable text representations behind one interface
    models          classifiers behind one interface (logreg, naive bayes, svm)
    analytics       dataset analysis: lengths, vocabulary, class balance
    evaluation      splitting, metrics, confusion analysis (dependency-free)
    training        cross-validation pipeline over embedding + classifier
    benchmarking    typed results, file registry, comparison utilities
    experiments     config, the central runner, and the baseline suite
    inference       a trained pipeline that turns raw text into a prediction
    explainability  top weighted features for linear models
    reporting       benchmark summaries (json/csv) and human-readable reports
    visualization   plots for datasets, evaluation, and comparisons
"""

__version__ = "0.1.0"
