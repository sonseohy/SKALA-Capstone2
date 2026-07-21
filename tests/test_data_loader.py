import pandas as pd
import pytest

from src.data_loader import RAW_PATH, clean_pandas, load_pandas, load_polars

pytestmark = pytest.mark.skipif(
    not RAW_PATH.exists(), reason="data/raw/adult.data 없음 (scripts/download_data.sh 먼저 실행)"
)


def test_load_pandas_shape():
    df = load_pandas()
    assert df.shape[1] == 15
    assert df.shape[0] > 30000


def test_clean_pandas_removes_duplicates():
    df = load_pandas()
    cleaned = clean_pandas(df)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) <= len(df)


def test_clean_pandas_removes_missing_values():
    df = load_pandas()
    cleaned = clean_pandas(df)
    assert cleaned.isna().sum().sum() == 0


def test_pandas_polars_row_count_match():
    pdf = load_pandas()
    pldf = load_polars()
    assert pdf.shape[0] == pldf.shape[0]


def test_missing_values_marked_as_na():
    df = load_pandas()
    assert df["workclass"].isna().sum() > 0
    assert isinstance(df, pd.DataFrame)
