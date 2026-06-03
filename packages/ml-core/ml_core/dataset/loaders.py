from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

import pandas as pd

from ml_core.dataset.dataset import TextDataset
from ml_core.dataset.schema import TextDatasetSchema


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "parquet support needs the 'parquet' extra: pip install 'itai-ml-core[parquet]'"
        ) from exc


# extension -> reader. adding json later is a one-line registration here.
_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": _load_csv,
    ".parquet": _load_parquet,
}


def supported_formats() -> list[str]:
    return sorted(_READERS)


def load_dataset(path: str | Path, schema: TextDatasetSchema) -> TextDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"unsupported format {path.suffix!r}; supported: {supported_formats()}"
        )
    return TextDataset(reader(path), schema)


def load_records(
    records: Sequence[Mapping[str, object]], schema: TextDatasetSchema
) -> TextDataset:
    # in-memory entry point used by the api so it never touches the filesystem.
    return TextDataset(pd.DataFrame(list(records)), schema)
