# build context is the repo root (see docker-compose.yml) so we can install
# the shared ml-core package alongside the api.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# install ml-core first so its layer caches independently of api changes
COPY packages/ml-core /packages/ml-core
RUN pip install --no-cache-dir -e /packages/ml-core

COPY apps/api/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

COPY apps/api ./

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
