"""ml-core: the machine learning library for the platform.

modules are organized by responsibility and kept loosely coupled so embeddings,
classifiers, and evaluation can be combined and benchmarked under identical conditions:

    preprocessing   configurable text cleaning and tokenization pipelines
    embeddings      interchangeable text representations behind one interface
    training        classifier training over any embedding
    evaluation      metrics, cross-validation, confusion-matrix analysis
    analytics       dataset analysis and quality scoring
    visualization   plots for datasets, embeddings, and results
"""

__version__ = "0.1.0"
