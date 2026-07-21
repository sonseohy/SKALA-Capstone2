"""Adult Census Income 모델 학습, 평가, 저장 모듈."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.model_preprocess import build_preprocessor

TARGET_COLUMN = "income"
TARGET_MAPPING = {
    "<=50K": 0,
    ">50K": 1,
}
DEFAULT_DATA_PATH = Path("data/processed/adult_clean.parquet")
DEFAULT_MODEL_PATH = Path("output/adult_income_pipeline.joblib")


def load_data(file_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """전처리 팀이 생성한 CSV 또는 Parquet 파일을 불러온다."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 Parquet을 사용하세요.")


def split_train_test(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """특성과 타깃을 분리하고 클래스 비율을 유지해 학습/평가셋으로 나눈다."""
    if target_column not in df.columns:
        raise ValueError(f"타깃 열이 없습니다: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype("string").str.strip().map(TARGET_MAPPING)

    if y.isna().any():
        invalid_values = df.loc[y.isna(), target_column].unique()
        raise ValueError(
            f"결측치 또는 알 수 없는 {target_column} 값이 있습니다: {invalid_values}"
        )

    return train_test_split(
        X,
        y.astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """정확도, F1 점수와 분류 리포트를 출력하고 주요 지표를 반환한다."""
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["<=50K", ">50K"],
            zero_division=0,
        )
    )
    return metrics


def save_model(model: Pipeline, file_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """학습이 끝난 전체 Pipeline을 joblib 파일로 저장한다."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def train_evaluate_save(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> tuple[Pipeline, dict]:
    """데이터 분리부터 모델 학습, 평가, 저장까지 순서대로 실행한다."""
    X_train, X_test, y_train, y_test = split_train_test(df)
    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    saved_path = save_model(model, model_path)
    print(f"\nModel saved: {saved_path}")
    return model, metrics


if __name__ == "__main__":
    dataset = load_data()
    train_evaluate_save(dataset)
