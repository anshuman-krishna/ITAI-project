# datasets

Dataset **metadata** and small reference files live here. Raw dataset files do **not** —
they're gitignored (everything except this README). Large files belong in object storage
(local volume now, S3/MinIO later); the database tracks them by metadata.

First dataset: the Kaggle "LLM — Detect AI Generated Text" competition data. Ingestion is
kept generic so new datasets drop in without architectural changes.
