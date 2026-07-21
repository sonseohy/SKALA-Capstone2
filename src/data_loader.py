"""Adult Census Income 데이터 로딩 유틸리티.

data/raw/adult.data (학습용) 를 Pandas / Polars 양쪽으로 읽어
동일한 스키마·타입으로 정리해 반환한다.
"""
import io
from pathlib import Path

import pandas as pd
import polars as pl

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week",
]


def load_pandas(path: Path = RAW_PATH) -> pd.DataFrame:
    """adult.data를 Pandas DataFrame으로 로딩. 결측치는 '?' -> NaN 처리.

    skipinitialspace=True가 값 선두 공백을 먼저 제거하므로,
    na_values는 공백 없는 '?' 로 지정해야 매칭된다.
    """
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    return df


def load_polars(path: Path = RAW_PATH) -> pl.DataFrame:
    """adult.data를 Polars DataFrame으로 로딩. 결측치는 '?' -> null 처리.

    원본 파일 끝에 빈 줄이 있어 그대로 읽으면 전부 null인 유령 행이
    생기고 숫자 컬럼 dtype 추론도 어긋나므로, 빈 줄을 먼저 제거하고
    숫자 컬럼 dtype을 명시적으로 고정한다.
    """
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    text = "\n".join(line for line in path.read_text().splitlines() if line.strip())
    df = pl.read_csv(
        io.StringIO(text),
        has_header=False,
        new_columns=COLUMNS,
        schema_overrides={c: pl.Int64 for c in NUMERIC_COLUMNS},
    ).with_columns(pl.col(pl.String).str.strip_chars())
    df = df.with_columns(
        pl.when(pl.col(c) == "?").then(None).otherwise(pl.col(c)).alias(c)
        for c in df.columns
        if df.schema[c] == pl.String
    )
    return df


def clean_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """문자열 컬럼 공백 제거 + 완전 중복행 제거 + 결측치 포함 행 제거."""
    obj_cols = df.select_dtypes(["object", "str"]).columns
    df = df.copy()
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[clean_pandas] 중복행 {removed}건 제거")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[clean_pandas] 결측치 포함 행 {removed}건 제거")

    return df


def basic_eda(df: pd.DataFrame) -> None:
    """콘솔에 기본 EDA 결과 출력: shape, dtypes, 결측치, 기술통계."""
    print("=" * 60)
    print(f"shape: {df.shape}")
    print("-" * 60)
    df.info()
    print("-" * 60)
    print("결측치 개수 (상위):")
    na = df.isna().sum()
    print(na[na > 0].sort_values(ascending=False))
    print("-" * 60)
    print(df.describe(include="all").T)
    print("=" * 60)


if __name__ == "__main__":
    pdf = clean_pandas(load_pandas())
    basic_eda(pdf)

    pldf = load_polars()
    print(f"\n[polars] shape: {pldf.shape}")
    print(pldf.null_count())
