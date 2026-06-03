## ITAI-project

## Overview

ITAI-project is a machine learning platform for detecting AI-generated text using Natural Language Processing and supervised learning techniques.

The project was developed to investigate how different text representations and machine learning algorithms perform when distinguishing between human-written and AI-generated content. In addition to the research component, the system was designed as a modular software platform that supports dataset analysis, experimentation, benchmarking, evaluation, and prediction workflows.

Although the current implementation focuses on AI-generated text detection, the architecture was intentionally designed to support future extensions such as authorship analysis, authenticity scoring, writing style assessment, and content verification systems.

---

## Objectives

The primary objectives of the project are:

* Detect AI-generated text using machine learning techniques
* Compare different feature extraction methods
* Benchmark multiple classification algorithms
* Build a reproducible experimentation framework
* Provide a complete prediction workflow
* Apply modern software engineering practices to machine learning systems

---

## Features

### Dataset Management

* Dataset ingestion
* Dataset validation
* Dataset profiling
* CSV support
* Parquet support
* Schema abstraction layer
* Duplicate detection
* Missing value analysis
* Class distribution analysis

### Text Preprocessing

* Lowercase normalization
* Punctuation removal
* Whitespace normalization
* Tokenization
* Stopword handling
* Lemmatization
* Configurable preprocessing pipelines

### Feature Extraction

* Bag-of-Words
* TF-IDF
* Shared embedding interface
* Sparse feature matrix support

### Machine Learning

* Logistic Regression
* Multinomial Naive Bayes
* Support Vector Machines
* Stratified Cross Validation
* Experiment Runner
* Benchmark Registry
* Model persistence

### Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix generation
* Benchmark comparison reports
* Experiment tracking

### Prediction System

* Text classification
* Confidence scoring
* Prediction API
* Stored model loading

### Software Platform

* FastAPI backend
* Next.js frontend
* Shared API contracts
* Dockerized services
* CI validation pipeline
* Modular monorepo architecture

---

## Technology Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy
* NLTK
* spaCy

### Backend

* FastAPI
* Pydantic
* SQLAlchemy

### Frontend

* Next.js
* TypeScript
* Tailwind CSS
* React Query

### Infrastructure

* Docker
* Docker Compose
* GitHub Actions
* PostgreSQL
* Redis
* Celery

---

## Repository Structure

```text
ITAI-project/

apps/
├── api/
├── web/
└── worker/

packages/
├── ml-core/
└── shared/

datasets/
docs/
reports/
notebooks/
scripts/
infrastructure/

testing/
```

### apps/api

Backend application built with FastAPI.

Responsibilities:

* Dataset APIs
* Benchmark APIs
* Prediction APIs
* Experiment execution
* Service orchestration

### apps/web

Frontend application built with Next.js.

Responsibilities:

* Dataset visualization
* Benchmark viewing
* Prediction interface
* User interaction

### apps/worker

Background processing service.

Responsibilities:

* Long-running experiments
* Report generation
* Future model training jobs

### packages/ml-core

Core machine learning library.

Contains:

* Dataset management
* Preprocessing
* Embeddings
* Training
* Evaluation
* Benchmarking
* Visualization

### packages/shared

Shared TypeScript contracts used across applications.

---

## Machine Learning Workflow

The system follows a structured machine learning pipeline:

```text
Dataset
   ↓
Validation
   ↓
Profiling
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Training
   ↓
Evaluation
   ↓
Benchmarking
   ↓
Prediction
```

Every experiment passes through the same workflow to ensure consistency and reproducibility.

---

## Dataset

The project was developed around the Kaggle dataset:

**LLM Detect AI Generated Text**

The system itself is dataset-agnostic and can support additional datasets as long as the required text and label fields are available.

Expected dataset structure:

```csv
text,generated
"sample text",0
"sample text",1
```

Where:

* `text` contains the document content
* `generated` contains the target label

---

## Running the Project

### Requirements

* Node.js 22+
* pnpm 10+
* Python 3.12+
* Docker
* Docker Compose

---

### Environment Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Update values as required.

---

### Start the Full Stack

```bash
make up
```

View logs:

```bash
make logs
```

Stop services:

```bash
make down
```

---

### Service URLs

Frontend:

```text
http://localhost:3000
```

Backend:

```text
http://localhost:8000
```

API Documentation:

```text
http://localhost:8000/docs
```

Health Endpoint:

```text
http://localhost:8000/api/v1/health
```

---

## Running Benchmarks

Example benchmark workflow:

```bash
python scripts/run_benchmark_suite.py \
  --csv datasets/train_essays.csv
```

Outputs include:

* Benchmark summaries
* Confusion matrices
* Evaluation reports
* Figures
* CSV exports
* JSON exports

Generated artifacts are stored in:

```text
reports/
docs/report-assets/
```

---

## API Overview

### Dataset Analysis

```http
POST /api/v1/datasets/profile
```

```http
POST /api/v1/datasets/validate
```

### Experiments

```http
POST /api/v1/experiments/run
```

```http
GET /api/v1/experiments/results
```

```http
GET /api/v1/experiments/compare
```

### Prediction

```http
POST /api/v1/predict
```

---

## Evaluation Strategy

Experiments are evaluated using:

* Stratified K-Fold Cross Validation
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix Analysis

This provides a more reliable assessment than relying on a single train-test split.

---

## Current Limitations

Several limitations should be considered when interpreting results:

* Dataset imbalance can affect evaluation metrics
* Classical embeddings do not fully capture semantic context
* Performance may vary across datasets
* AI-generated text quality continues to improve rapidly

These limitations are discussed in detail within the project report.

---

## Future Work

Potential future extensions include:

* Word2Vec integration
* Doc2Vec integration
* Transformer embeddings
* Sentence Transformers
* XGBoost classifiers
* Explainable AI techniques
* Multilingual detection
* Cloud deployment
* User authentication
* Advanced analytics dashboards

The current architecture was designed to support these additions without major structural changes.

---

## Development Notes

The repository follows a modular architecture with strong separation of concerns.

Key design principles:

* Reproducibility
* Maintainability
* Extensibility
* Clear interfaces
* Minimal coupling

Machine learning logic remains independent from frontend and API layers, allowing experimentation workflows to evolve without affecting application infrastructure.

---

## License

This project was developed for academic and research purposes.
