# build context is the repo root (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY packages/ml-core /packages/ml-core
RUN pip install --no-cache-dir -e /packages/ml-core

COPY apps/worker/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

COPY apps/worker ./

CMD ["celery", "-A", "app.celery_app.celery_app", "worker", "--loglevel=info"]
