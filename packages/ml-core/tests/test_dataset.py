import pandas as pd
import pytest

from ml_core.dataset import (
    TextDataset,
    TextDatasetSchema,
    load_dataset,
    load_records,
    profile_dataset,
    validate_dataset,
)

SCHEMA = TextDatasetSchema(text_column="content", label_column="generated")


def _records():
    return [
        {"content": "humans write naturally and at length", "generated": 0},
        {"content": "ai produces fluent text", "generated": 1},
        {"content": "another human sentence here", "generated": 0},
        {"content": "more generated output appears", "generated": 1},
    ]


def test_load_csv_roundtrip(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(_records()).to_csv(path, index=False)
    dataset = load_dataset(path, SCHEMA)
    assert len(dataset) == 4
    assert dataset.texts[0].startswith("humans")
    assert dataset.labels == [0, 1, 0, 1]


def test_schema_missing_column_raises():
    with pytest.raises(ValueError):
        TextDataset(pd.DataFrame({"text": ["x"]}), SCHEMA)


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope")
    with pytest.raises(ValueError):
        load_dataset(path, SCHEMA)


def test_profile_counts_and_distribution():
    dataset = load_records(_records(), SCHEMA)
    profile = profile_dataset(dataset)
    assert profile.total_samples == 4
    assert profile.class_distribution is not None
    assert profile.class_distribution.counts == {"0": 2, "1": 2}
    assert profile.class_distribution.imbalance_ratio == 1.0
    assert profile.word_length.maximum >= profile.word_length.minimum
    assert profile.vocabulary.size > 0


def test_validate_flags_empty_and_duplicates():
    records = _records() + [{"content": "   ", "generated": 0}, _records()[0]]
    dataset = load_records(records, SCHEMA)
    report = validate_dataset(dataset)
    codes = {i.code for i in report.issues}
    assert "empty_text" in codes
    assert "duplicate_rows" in codes
    assert report.is_valid is False  # empty text is an error


def test_validate_invalid_labels():
    records = [{"content": "a", "generated": 0}, {"content": "b", "generated": 9}]
    report = validate_dataset(load_records(records, SCHEMA), allowed_labels=[0, 1])
    assert any(i.code == "invalid_label" for i in report.issues)
    assert report.is_valid is False


def test_validate_clean_dataset_passes():
    report = validate_dataset(load_records(_records(), SCHEMA), allowed_labels=[0, 1])
    assert report.is_valid is True
    assert report.issues == []
