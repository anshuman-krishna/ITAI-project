"""dataset intelligence layer: ingestion, schema, validation, profiling.

a raw table becomes a canonical TextDataset described by a TextDatasetSchema. from there
it can be profiled (typed DatasetProfile) and validated (typed ValidationReport) without
column assumptions or dataframes leaking into the rest of the system.
"""

from ml_core.dataset.dataset import TextDataset
from ml_core.dataset.loaders import (
    load_dataset,
    load_records,
    supported_formats,
)
from ml_core.dataset.profilers import profile_dataset
from ml_core.dataset.schema import TextDatasetSchema
from ml_core.dataset.types import (
    DatasetProfile,
    ValidationIssue,
    ValidationReport,
)
from ml_core.dataset.validators import validate_dataset

__all__ = [
    "TextDataset",
    "TextDatasetSchema",
    "load_dataset",
    "load_records",
    "supported_formats",
    "profile_dataset",
    "validate_dataset",
    "DatasetProfile",
    "ValidationReport",
    "ValidationIssue",
]
