"""
작성일: 2026-07-21
작성자: 신동운

작성 목적:
앞 단계에서 전처리한 Adult Census Income 데이터를 이용해 개인 특성과 소득
구간의 관계를 기술통계와 통계 검정으로 분석하기 위해 작성했다.

프로그램 설명:
앞 단계에서 중복 행과 결측 행을 제거해 저장한 `adult_clean.parquet`을 불러온다.
수치형 변수의 기술통계와 상관계수를 구하고, 소득 집단별 t-test, 근로 형태별
ANOVA, 범주형 변수의 카이제곱 검정과 Cramér's V를 계산한다. 분석 결과는
별도의 statistical_analysis_report.md 파일로 저장한다.

주요 분석 변수:
- income: 연간 소득이 50,000달러 이하인지 초과인지 나타내는 목표 변수
- income_binary: 소득 초과는 1, 이하는 0으로 변환한 모델 학습용 목표 변수
- age: 나이
- education_num: 학력 수준을 숫자로 나타낸 값
- hours_per_week: 주당 근무 시간
- workclass: 근로 형태 또는 고용 부문
- occupation: 직업군

프로그램 요약:
1. 앞 단계에서 전처리한 데이터를 불러오고 소득 목표 변수를 0과 1로 변환한다.
2. y가 0인 집단과 1인 집단의 크기와 주요 특성을 비교한다.
3. 성별, 근로 형태, 학력별 y=1 비율을 구한다.
4. 평균, 표준편차, 분위수와 피어슨 상관계수를 구한다.
5. 정규성과 등분산성을 확인한 뒤 소득 집단별 평균을 비교한다.
6. 여러 근로 형태의 평균을 ANOVA로 비교하고 비모수 검정도 함께 확인한다.
7. 범주형 변수의 연관성을 카이제곱 검정과 Cramér's V로 확인한다.

변경 내역:
- 2026-07-21: 데이터 전처리와 기본 기술통계 및 t-test를 작성했다.
- 2026-07-21: 정규성·등분산 검정, ANOVA, 카이제곱 검정, Cramér's V와
  Markdown 보고서 생성을 추가했다.
- 2026-07-21: 앞 단계와 겹치는 Pandas·Polars 전처리를 제거하고 전처리된
  Parquet 파일을 불러오도록 변경했다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    levene,
    normaltest,
    ttest_ind,
)

from .data_loader import NUMERIC_COLUMNS

ALPHA = 0.05
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data/processed/adult_clean.parquet"
REPORT_PATH = Path(__file__).resolve().parent.parent / "statistical_analysis_report.md"

T_TEST_VARIABLES = {
    "age": "나이",
    "fnlwgt": "표본 가중치",
    "education_num": "학력 수준",
    "capital_gain": "자본 이득",
    "capital_loss": "자본 손실",
    "hours_per_week": "주당 근무 시간",
}

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

CATEGORICAL_DESCRIPTIONS = {
    "workclass": "근로 형태",
    "education": "학력",
    "marital_status": "혼인 상태",
    "occupation": "직업",
    "relationship": "가구 내 관계",
    "race": "인종",
    "sex": "성별",
    "native_country": "출신 국가",
}


def format_p_value(p_value: float) -> str:
    """p-value를 읽기 쉬운 문자열로 바꾼다."""
    if p_value == 0:
        return "< 1e-300"
    if p_value < 0.0001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def load_analysis_data(path: Path = PROCESSED_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    """앞 단계에서 전처리해 저장한 데이터를 불러온다."""
    if not path.exists():
        raise FileNotFoundError(
            "전처리 데이터가 없습니다. 먼저 데이터 전처리 코드를 실행하세요: "
            f"{path}"
        )

    df = pd.read_parquet(path)
    duplicate_count = int(df.duplicated().sum())
    missing_count = int(df.isna().sum().sum())

    if duplicate_count or missing_count:
        raise ValueError(
            "전처리 데이터에 중복 행 또는 결측값이 남아 있습니다. "
            "앞 단계의 전처리 결과를 확인하세요."
        )

    source_columns = df.shape[1]
    income_values = set(df["income"].unique())
    if not income_values.issubset({"<=50K", ">50K"}):
        raise ValueError(f"알 수 없는 income 값이 있습니다: {income_values}")

    # 원래 소득 구간은 남겨두고 모델 학습에 사용할 0과 1 형태의 변수를 추가한다.
    df["income_binary"] = (df["income"] == ">50K").astype(int)

    summary = {
        "rows": len(df),
        "source_columns": source_columns,
        "analysis_columns": df.shape[1],
        "duplicates": duplicate_count,
        "missing_values": missing_count,
    }

    print("[분석 데이터 불러오기]")
    print(f"전처리 데이터: {summary['rows']:,}행, {summary['source_columns']}열")
    print("income_binary 변환: >50K는 1, <=50K는 0")
    print(f"분석 데이터: {summary['rows']:,}행, {summary['analysis_columns']}열\n")

    return df, summary


def calculate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 변수의 평균, 표준편차, 최솟값, 분위수, 최댓값을 구한다."""
    statistics = df[NUMERIC_COLUMNS].describe().T
    return statistics[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]


def calculate_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """y가 0인 집단과 1인 집단의 주요 수치형 기술통계를 구한다."""
    return df.groupby("income_binary")[list(T_TEST_VARIABLES)].agg(
        ["count", "mean", "std", "median"]
    )


def calculate_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """y=0과 y=1의 개수와 비율을 구한다."""
    distribution = df["income_binary"].value_counts().sort_index().to_frame("count")
    distribution["ratio_percent"] = distribution["count"] / len(df) * 100
    distribution.index.name = "income_binary"
    return distribution


def calculate_positive_rates(df: pd.DataFrame) -> pd.DataFrame:
    """성별, 근로 형태, 학력별로 y=1의 개수와 비율을 구한다."""
    results = []

    for column in ["sex", "workclass", "education"]:
        summary = (
            df.groupby(column)["income_binary"]
            .agg(total_count="count", y1_count="sum", y1_ratio="mean")
            .reset_index()
            .rename(columns={column: "group_value"})
        )
        summary.insert(0, "group_variable", column)
        summary["y1_ratio_percent"] = summary.pop("y1_ratio") * 100
        results.append(summary)

    return pd.concat(results, ignore_index=True)


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 변수 사이의 피어슨 상관계수를 구한다."""
    correlation_columns = [*NUMERIC_COLUMNS, "income_binary"]
    return df[correlation_columns].corr(method="pearson")


def perform_income_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    """소득 두 집단의 주요 수치형 변수 평균을 독립표본 t-test로 비교한다."""
    records = []

    for variable, description in T_TEST_VARIABLES.items():
        low_income = df.loc[df["income_binary"] == 0, variable]
        high_income = df.loc[df["income_binary"] == 1, variable]

        low_normality = normaltest(low_income)
        high_normality = normaltest(high_income)
        variance_test = levene(low_income, high_income, center="median")
        equal_variance = variance_test.pvalue >= ALPHA

        test_result = ttest_ind(
            low_income,
            high_income,
            equal_var=equal_variance,
        )

        records.append(
            {
                "variable": variable,
                "description": description,
                "y0_mean": low_income.mean(),
                "y1_mean": high_income.mean(),
                "normality_p_low": low_normality.pvalue,
                "normality_p_high": high_normality.pvalue,
                "levene_p": variance_test.pvalue,
                "test_type": "Student t-test" if equal_variance else "Welch t-test",
                "t_statistic": test_result.statistic,
                "p_value": test_result.pvalue,
                "significant": test_result.pvalue < ALPHA,
            }
        )

    return pd.DataFrame(records)


def perform_workclass_anova(df: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    """근로 형태별 주당 근무 시간 평균 차이를 ANOVA로 검정한다."""
    grouped = [
        (name, group["hours_per_week"].to_numpy())
        for name, group in df.groupby("workclass")
    ]
    group_names = [name for name, _ in grouped]
    groups = [values for _, values in grouped]

    normality_p_values = [normaltest(values).pvalue for values in groups]
    variance_test = levene(*groups, center="median")
    anova_result = f_oneway(*groups)
    kruskal_result = kruskal(*groups)

    group_summary = (
        df.groupby("workclass")["hours_per_week"]
        .agg(["count", "mean", "std", "median"])
        .loc[group_names]
        .sort_values("mean", ascending=False)
    )

    result = {
        "group_count": len(groups),
        "normal_group_count": sum(p >= ALPHA for p in normality_p_values),
        "levene_statistic": float(variance_test.statistic),
        "levene_p": float(variance_test.pvalue),
        "anova_statistic": float(anova_result.statistic),
        "anova_p": float(anova_result.pvalue),
        "kruskal_statistic": float(kruskal_result.statistic),
        "kruskal_p": float(kruskal_result.pvalue),
    }
    return result, group_summary


def cramers_v(first: pd.Series, second: pd.Series) -> float:
    """두 범주형 변수의 연관성 크기를 보정된 Cramér's V로 계산한다."""
    table = pd.crosstab(first, second)
    chi2 = chi2_contingency(table, correction=False).statistic
    n = table.to_numpy().sum()
    rows, columns = table.shape

    phi2 = chi2 / n
    corrected_phi2 = max(0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    corrected_rows = rows - ((rows - 1) ** 2) / (n - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)

    return float(np.sqrt(corrected_phi2 / denominator)) if denominator > 0 else 0.0


def calculate_categorical_associations(df: pd.DataFrame) -> pd.DataFrame:
    """각 범주형 변수와 소득 구간의 관계를 카이제곱 검정으로 확인한다."""
    records = []

    for column in CATEGORICAL_COLUMNS:
        if column == "income":
            continue

        table = pd.crosstab(df[column], df["income"])
        test_result = chi2_contingency(table)
        expected = test_result.expected_freq
        expected_under_five = int((expected < 5).sum())
        expected_ratio = expected_under_five / expected.size * 100

        records.append(
            {
                "variable": column,
                "chi2": test_result.statistic,
                "p_value": test_result.pvalue,
                "cramers_v": cramers_v(df[column], df["income"]),
                "expected_under_5": expected_under_five,
                "expected_under_5_ratio": expected_ratio,
                "assumption_met": expected_ratio < 20,
            }
        )

    return pd.DataFrame(records).sort_values("cramers_v", ascending=False)


def calculate_cramers_v_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """모든 범주형 변수 조합의 Cramér's V 행렬을 만든다."""
    matrix = pd.DataFrame(
        np.eye(len(CATEGORICAL_COLUMNS)),
        index=CATEGORICAL_COLUMNS,
        columns=CATEGORICAL_COLUMNS,
    )

    for first_index, first in enumerate(CATEGORICAL_COLUMNS):
        for second in CATEGORICAL_COLUMNS[first_index + 1 :]:
            association = cramers_v(df[first], df[second])
            matrix.loc[first, second] = association
            matrix.loc[second, first] = association

    return matrix


def dataframe_block(df: pd.DataFrame, decimals: int = 3) -> str:
    """DataFrame을 Markdown 코드 블록 안에 넣을 문자열로 바꾼다."""
    table = "\n".join(
        line.rstrip() for line in df.round(decimals).to_string().splitlines()
    )
    return f"```text\n{table}\n```"


def summarize_significance(
    t_tests: pd.DataFrame,
    categorical_results: pd.DataFrame,
) -> dict[str, list[str]]:
    """검정 결과를 유의한 변수와 유의하지 않은 변수로 나눈다."""
    significant_numeric = t_tests.loc[t_tests["significant"], "description"].tolist()
    non_significant_numeric = t_tests.loc[
        ~t_tests["significant"], "description"
    ].tolist()
    significant_categorical = categorical_results.loc[
        categorical_results["p_value"] < ALPHA, "variable"
    ].tolist()
    non_significant_categorical = categorical_results.loc[
        categorical_results["p_value"] >= ALPHA, "variable"
    ].tolist()
    significant_categorical = [
        CATEGORICAL_DESCRIPTIONS[name] for name in significant_categorical
    ]
    non_significant_categorical = [
        CATEGORICAL_DESCRIPTIONS[name] for name in non_significant_categorical
    ]

    return {
        "significant_numeric": significant_numeric,
        "non_significant_numeric": non_significant_numeric,
        "significant_categorical": significant_categorical,
        "non_significant_categorical": non_significant_categorical,
    }


def join_variable_names(names: list[str]) -> str:
    """변수 이름 목록을 보고서에 넣을 문자열로 바꾼다."""
    return ", ".join(names) if names else "없음"


def create_report(
    data_summary: dict[str, int],
    descriptive: pd.DataFrame,
    group_statistics: pd.DataFrame,
    target_distribution: pd.DataFrame,
    positive_rates: pd.DataFrame,
    correlations: pd.DataFrame,
    t_tests: pd.DataFrame,
    anova_result: dict[str, float | int],
    workclass_summary: pd.DataFrame,
    categorical_results: pd.DataFrame,
    cramers_matrix: pd.DataFrame,
) -> None:
    """통계 분석 결과를 Markdown 보고서로 저장한다."""
    t_test_table = t_tests[
        [
            "description",
            "y0_mean",
            "y1_mean",
            "normality_p_low",
            "normality_p_high",
            "levene_p",
            "test_type",
            "t_statistic",
            "p_value",
            "significant",
        ]
    ].copy()
    for column in ["normality_p_low", "normality_p_high", "levene_p", "p_value"]:
        t_test_table[column] = t_test_table[column].map(format_p_value)

    categorical_table = categorical_results[
        [
            "variable",
            "chi2",
            "p_value",
            "cramers_v",
            "expected_under_5",
            "expected_under_5_ratio",
            "assumption_met",
        ]
    ].copy()
    categorical_table["p_value"] = categorical_table["p_value"].map(format_p_value)

    strongest_category = categorical_results.iloc[0]
    strongest_category_name = CATEGORICAL_DESCRIPTIONS[
        str(strongest_category["variable"])
    ]
    significance = summarize_significance(t_tests, categorical_results)
    t_test_conclusions = "\n".join(
        f"- {row['description']}: p-value {format_p_value(row['p_value'])}, "
        + (
            "두 소득 집단의 평균에 유의한 차이가 있다."
            if row["significant"]
            else "두 소득 집단의 평균에 유의한 차이가 있다고 보기 어렵다."
        )
        for _, row in t_tests.iterrows()
    )
    y0_count = int(target_distribution.loc[0, "count"])
    y1_count = int(target_distribution.loc[1, "count"])
    y0_ratio = float(target_distribution.loc[0, "ratio_percent"])
    y1_ratio = float(target_distribution.loc[1, "ratio_percent"])
    unequal_variance_variables = t_tests.loc[
        t_tests["levene_p"] < ALPHA, "description"
    ].tolist()

    report = f"""# Adult Census Income 통계 분석 결과

## 분석 결과 요약

- y=0은 {y0_count:,}명({y0_ratio:.1f}%), y=1은 {y1_count:,}명({y1_ratio:.1f}%)으로 y=0이 더 많았다.
- 유의한 수치형 변수: {join_variable_names(significance['significant_numeric'])}
- 유의하지 않은 수치형 변수: {join_variable_names(significance['non_significant_numeric'])}
- 유의한 범주형 변수: {join_variable_names(significance['significant_categorical'])}
- 유의하지 않은 범주형 변수: {join_variable_names(significance['non_significant_categorical'])}
- 범주형 변수 중 소득과 가장 강한 관계를 보인 변수는
  {strongest_category_name}이고 Cramér's V는 {strongest_category['cramers_v']:.3f}이었다.
- 근로 형태별 주당 근무 시간은 통계적으로 유의한 차이가 있었다.

유의한 변수와 유의하지 않은 변수를 빠르게 확인할 수 있도록 주요 결과를 먼저
정리했다. 이 결과만으로 머신러닝 입력 변수를 바로 결정한 것은 아니며, 실제
변수 선택은 모델 성능과 변수 간 관계도 함께 확인해야 한다.

## 1. 프로그램 설명

Adult 데이터에서 중복 행과 결측 행을 제거한 뒤 기술통계, 상관분석, 집단별
평균 비교와 범주형 변수의 연관성 분석을 진행했다. 유의수준은 0.05로 정했다.

## 2. 분석 데이터 준비

앞 단계에서 중복 행과 결측 행을 제거해 저장한 데이터를 불러왔다. 통계 분석
파일에서는 같은 전처리를 반복하지 않고 저장된 결과를 사용했다.

- 불러온 전처리 데이터: {data_summary['rows']:,}행, {data_summary['source_columns']}열
- 데이터 확인 결과 중복 행: {data_summary['duplicates']}건
- 데이터 확인 결과 결측값: {data_summary['missing_values']}개
- 최종 분석 데이터: {data_summary['rows']:,}행, {data_summary['analysis_columns']}열
- 소득 목표 변수 변환: `>50K`는 1, `<=50K`는 0인 `income_binary` 추가

## 3. 목표 변수 y의 분포

`income_binary`를 모델의 목표 변수 y로 사용했다. y=0은 연 소득 50,000달러
이하이고, y=1은 50,000달러 초과다. 각 집단의 크기를 먼저 확인해 목표 변수의
불균형 정도를 확인했다.

{dataframe_block(target_distribution, 2)}

## 4. 집단별 y=1 비율

성별, 근로 형태, 학력별로 전체 인원 중 y=1인 인원과 비율을 계산했다. 이는
집단마다 전체 인원수가 다르기 때문에 y=1인 인원수만 비교하지 않고 비율도
함께 확인한 것이다.

{dataframe_block(positive_rates, 2)}

## 5. 전체 기술통계

{dataframe_block(descriptive, 2)}

## 6. y=0과 y=1 집단별 기술통계

{dataframe_block(group_statistics, 2)}

## 7. 수치형 변수의 피어슨 상관계수

상관계수는 -1에서 1 사이의 값이며, 절댓값이 클수록 두 수치형 변수의 선형
관계가 강하다는 뜻이다. `income_binary`를 포함해 수치형 변수와 소득 구간의
관계도 함께 확인했다.

{dataframe_block(correlations, 3)}

## 8. y=0과 y=1 집단별 독립표본 t-test

- H0(귀무가설): 두 소득 집단의 해당 변수 평균은 같다.
- H1(대립가설): 두 소득 집단의 해당 변수 평균은 다르다.
- 정규성 검정: D'Agostino의 normaltest를 사용했다.
- 등분산 검정: Levene 검정을 사용했다.
- 등분산성이 충족되면 Student t-test, 충족되지 않으면 Welch t-test를 사용했다.

{dataframe_block(t_test_table, 4)}

Levene 검정 결과 {join_variable_names(unequal_variance_variables)}의 p-value가
모두 0.05보다 작아 등분산 가정이 충족되지 않았다. 따라서 해당 변수는 모두
Welch t-test로 평균 차이를 검정했다.

표본 수가 많아 정규성 검정은 작은 분포 차이에도 민감하다. 따라서 정규성
검정 결과만으로 분석을 중단하지 않고, 등분산 결과에 따라 t-test 종류를
선택해 평균 차이를 확인했다.

{t_test_conclusions}

## 9. 근로 형태별 주당 근무 시간 ANOVA

- H0(귀무가설): 모든 근로 형태의 평균 주당 근무 시간은 같다.
- H1(대립가설): 하나 이상의 근로 형태에서 평균 주당 근무 시간이 다르다.
- 비교 집단: {anova_result['group_count']}개
- 정규성 검정을 통과한 집단: {anova_result['normal_group_count']}개
- Levene p-value: {format_p_value(float(anova_result['levene_p']))}
- ANOVA p-value: {format_p_value(float(anova_result['anova_p']))}
- Kruskal-Wallis p-value: {format_p_value(float(anova_result['kruskal_p']))}

{dataframe_block(workclass_summary, 2)}

Levene 검정의 p-value가 0.05보다 작으면 등분산 가정이 충족되지 않은 것이다.
이 경우 일반 ANOVA만으로 결론을 내리지 않고, 분포 가정이 더 적은
Kruskal-Wallis 검정 결과도 함께 확인했다. 두 검정 모두 유의하면 근로 형태에
따라 주당 근무 시간에 차이가 있다고 해석할 수 있다. 어느 집단끼리 차이가
있는지 확인하려면 별도의 사후검정이 필요하다.

## 10. 범주형 변수와 소득 구간의 관계

카이제곱 검정으로 두 범주형 변수의 독립성을 확인하고, Cramér's V로 관계의
크기를 확인했다. Cramér's V는 0에 가까울수록 관계가 약하고 1에 가까울수록
강하다. 기대빈도가 5 미만인 셀이 전체의 20% 미만인지도 확인했다.

{dataframe_block(categorical_table, 4)}

소득 구간과 가장 강한 관계를 보인 범주형 변수는
`{strongest_category['variable']}`이고 Cramér's V는
{strongest_category['cramers_v']:.3f}이다.

## 11. 범주형 변수 간 Cramér's V 행렬

{dataframe_block(cramers_matrix, 3)}

## 12. 통계적 유의성에 따른 변수 정리

유의수준 0.05를 기준으로 검정한 변수를 다음과 같이 나눴다.

- 유의한 수치형 변수: {join_variable_names(significance['significant_numeric'])}
- 유의하지 않은 수치형 변수: {join_variable_names(significance['non_significant_numeric'])}
- 유의한 범주형 변수: {join_variable_names(significance['significant_categorical'])}
- 유의하지 않은 범주형 변수: {join_variable_names(significance['non_significant_categorical'])}

표본 수가 많으면 작은 차이도 통계적으로 유의하게 나올 수 있으므로 p-value만
보지 않고 평균 차이, 상관계수와 Cramér's V도 함께 확인했다.

## 13. 분석 요약

- 전처리 후 결측값과 완전 중복 행이 없는 30,139건을 분석에 사용했다.
- 원래 소득 구간은 보존하고 모델 학습용 `income_binary` 변수를 추가했다.
- y=0과 y=1의 개수와 비율, 성별·근로 형태·학력별 y=1 비율을 비교했다.
- 소득 집단에 따라 전체 수치형 변수의 평균 차이를 검정했고, 표본 가중치만
  유의한 차이가 나오지 않았다.
- 근로 형태별 주당 근무 시간 차이는 ANOVA와 Kruskal-Wallis 검정으로 확인했다.
- 범주형 변수는 카이제곱 검정과 Cramér's V를 사용해 소득 구간과의 관계를 확인했다.
- 통계적으로 유의한 관계가 있더라도 해당 변수가 소득의 원인이라는 뜻은 아니다.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n통계 분석 보고서 저장 완료: {REPORT_PATH.name}")


def main() -> None:
    """전처리 데이터를 불러와 통계 분석과 보고서 생성을 실행한다."""
    pandas_df, data_summary = load_analysis_data()

    descriptive = calculate_descriptive_statistics(pandas_df)
    group_statistics = calculate_group_statistics(pandas_df)
    target_distribution = calculate_target_distribution(pandas_df)
    positive_rates = calculate_positive_rates(pandas_df)
    correlations = calculate_correlations(pandas_df)
    t_tests = perform_income_t_tests(pandas_df)
    anova_result, workclass_summary = perform_workclass_anova(pandas_df)
    categorical_results = calculate_categorical_associations(pandas_df)
    cramers_matrix = calculate_cramers_v_matrix(pandas_df)

    print("[기술통계]")
    print(descriptive.round(2))

    print("\n[y=0과 y=1 분포]")
    print(target_distribution.round(2))

    print("\n[집단별 y=1 비율]")
    print(positive_rates.round(2))

    print("\n[수치형 변수 상관계수]")
    print(correlations.round(3))

    print("\n[소득 집단별 독립표본 t-test]")
    t_test_console = t_tests[
        ["description", "test_type", "t_statistic", "p_value"]
    ].copy()
    t_test_console["p_value"] = t_test_console["p_value"].map(format_p_value)
    print(t_test_console)

    print("\n[근로 형태별 주당 근무 시간 ANOVA]")
    print(
        f"ANOVA p-value: {format_p_value(float(anova_result['anova_p']))}, "
        f"Kruskal-Wallis p-value: "
        f"{format_p_value(float(anova_result['kruskal_p']))}"
    )

    print("\n[범주형 변수와 소득 구간의 관계]")
    categorical_console = categorical_results[
        ["variable", "p_value", "cramers_v"]
    ].copy()
    categorical_console["p_value"] = categorical_console["p_value"].map(
        format_p_value
    )
    print(categorical_console.round(4))

    significance = summarize_significance(t_tests, categorical_results)
    print("\n[유의한 변수와 유의하지 않은 변수]")
    print(
        "유의한 수치형 변수: "
        f"{join_variable_names(significance['significant_numeric'])}"
    )
    print(
        "유의하지 않은 수치형 변수: "
        f"{join_variable_names(significance['non_significant_numeric'])}"
    )
    print(
        "유의한 범주형 변수: "
        f"{join_variable_names(significance['significant_categorical'])}"
    )
    print(
        "유의하지 않은 범주형 변수: "
        f"{join_variable_names(significance['non_significant_categorical'])}"
    )

    create_report(
        data_summary,
        descriptive,
        group_statistics,
        target_distribution,
        positive_rates,
        correlations,
        t_tests,
        anova_result,
        workclass_summary,
        categorical_results,
        cramers_matrix,
    )


if __name__ == "__main__":
    main()
