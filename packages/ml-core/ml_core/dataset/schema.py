from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextDatasetSchema:
    """maps the columns of a raw table to the platform's canonical roles.

    every dataset is described by this schema instead of column-name assumptions
    leaking across loaders, profilers, validators, and embeddings. new datasets are
    supported by supplying a schema, not by changing code.
    """

    text_column: str
    label_column: str | None = None
    metadata_columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_labels(self) -> bool:
        return self.label_column is not None

    def required_columns(self) -> list[str]:
        cols = [self.text_column]
        if self.label_column is not None:
            cols.append(self.label_column)
        cols.extend(self.metadata_columns)
        return cols
