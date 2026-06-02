# worker

Celery worker for jobs that are too slow to run inside an API request: model training,
embedding generation, report generation, and bulk prediction. Redis is the broker.

```
app/
  celery_app.py   celery instance and config
  tasks/          task modules (registered via autodiscovery)
```

## Run locally

```bash
pip install -e ../../packages/ml-core
pip install -e ".[dev]"
celery -A app.celery_app.celery_app worker --loglevel=info
```

No real tasks exist yet — this is the foundation for Phase 2+.
