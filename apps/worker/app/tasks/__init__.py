"""celery tasks.

planned job types: model training, embedding generation, report generation,
and bulk prediction. these run here because they are too slow for a request cycle.
each gets its own module and is decorated with @celery_app.task.
"""
