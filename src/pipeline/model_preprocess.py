"""Adult Census Income 데이터 전처리 파이프라인 모듈."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """수치형/범주형 컬럼을 구분하여 결측치 처리 및 변환(Scale/OneHot) 수행."""
    # 1. 수치형 컬럼과 범주형 컬럼 자동 구분
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "category", "string"]).columns

    # 2. 수치형 전처리 (중앙값 결측치 채우기 + 표준화)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 3. 범주형 전처리 (최빈값 결측치 채우기 + 원-핫 인코딩)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 4. ColumnTransformer로 결합
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor