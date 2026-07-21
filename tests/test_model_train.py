import pandas as pd

from src.pipeline.model_train import (
    build_model_pipeline,
    save_model,
    split_train_test,
)


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 28, 48],
            "hours_per_week": [40, 35, 45, 50, 40, 55, 30, 60, 42, 38],
            "workclass": [
                "Private",
                "Private",
                "Self-emp",
                "Private",
                None,
                "Self-emp",
                "Private",
                "Self-emp",
                "Private",
                "Private",
            ],
            "income": [
                "<=50K",
                "<=50K",
                ">50K",
                ">50K",
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
            ],
        }
    )


def test_split_train_test_preserves_all_rows():
    df = sample_data()

    X_train, X_test, y_train, y_test = split_train_test(df)

    assert len(X_train) + len(X_test) == len(df)
    assert "income" not in X_train.columns
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_pipeline_trains_predicts_and_saves(tmp_path):
    X_train, X_test, y_train, _ = split_train_test(sample_data())
    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    saved_path = save_model(model, tmp_path / "model.joblib")

    assert len(predictions) == len(X_test)
    assert saved_path.exists()
