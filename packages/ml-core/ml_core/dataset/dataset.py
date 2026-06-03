from __future__ import annotations

import pandas as pd

from ml_core.dataset.schema import TextDatasetSchema


class TextDataset:
    """canonical in-memory dataset.

    wraps a dataframe but does not let it leak: external code (api, services) works
    through texts, labels, and typed result objects. the underlying frame is available
    to ml-core internals via `frame` for column-level checks, and nowhere else.
    """

    def __init__(self, frame: pd.DataFrame, schema: TextDatasetSchema) -> None:
        missing = [c for c in schema.required_columns() if c not in frame.columns]
        if missing:
            raise ValueError(f"columns missing for schema: {missing}")
        self._frame = frame.reset_index(drop=True)
        self._schema = schema

    @property
    def schema(self) -> TextDatasetSchema:
        return self._schema

    @property
    def frame(self) -> pd.DataFrame:
        # ml-core-internal accessor. do not return this across package boundaries.
        return self._frame

    @property
    def columns(self) -> list[str]:
        return list(self._frame.columns)

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def texts(self) -> list[str]:
        # null texts are surfaced as empty strings so downstream code never sees nan.
        return self._frame[self._schema.text_column].fillna("").astype(str).tolist()

    @property
    def labels(self) -> list[object] | None:
        if not self._schema.has_labels:
            return None
        return self._frame[self._schema.label_column].tolist()

    def column(self, name: str) -> list[object]:
        if name not in self._frame.columns:
            raise KeyError(name)
        return self._frame[name].tolist()
