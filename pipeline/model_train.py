import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_train_test(df):
    X = df.drop(columns=['income'])
    Y = df['income']

    target_mapping = {
        "<=50K": 0,
        ">50K": 1,
    }

    y = Y.str.strip().map(target_mapping)

    if y.isna().any():
        invalid_values = df.loc[y.isna(), "income"].unique()
        raise ValueError(f"알 수 없는 income 값이 있습니다: {invalid_values}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

