"""Seaborn 정적 차트 유틸리티.

data/processed/adult_clean.parquet (또는 src.data_loader 결과)을 입력으로 받아
차트를 그리고 output/ 에 저장한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


"""
나이(age) 분포를 소득(income) 그룹별로 비교하는 Histogram + KDE.

고소득층(>50K)과 저소득층(<=50K)의 나이대 분포 차이를 확인한다.
"""
def plot_age_by_income(df: pd.DataFrame, save: bool = True) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(8, 6))
    order = sorted(df["income"].unique())
    sns.histplot(
        data=df, x="age", hue="income", hue_order=order,
        kde=True, stat="density", common_norm=False,
        palette="Set2", alpha=0.5, ax=ax,
    )
    ax.set_title("나이 분포 (소득별) (Age Distribution by Income)")
    ax.set_xlabel("나이 (Age)")
    ax.set_ylabel("밀도 (Density)")
    ax.get_legend().set_title("소득")

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUT_DIR / "chart2_age_by_income.png", dpi=150, bbox_inches="tight")
        print(f"[plot_age_by_income] 저장됨: {OUTPUT_DIR / 'chart2_age_by_income.png'}")

    return ax


"""
수치형 변수 간 상관관계 Heatmap.

age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
간의 피어슨 상관계수를 한눈에 비교한다.
"""
def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Axes:
    from src.data_loader import NUMERIC_COLUMNS

    corr = df[NUMERIC_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("수치형 변수 간 상관관계 (Correlation Heatmap)")

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUT_DIR / "chart_correlation_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"[plot_correlation_heatmap] 저장됨: {OUTPUT_DIR / 'chart_correlation_heatmap.png'}")

    return ax


"""
근무시간(hours_per_week) vs 소득(income) Box plot.

고소득층(>50K)이 저소득층(<=50K)보다 더 오래 일하는지,
이상치·중앙값 분포를 함께 확인한다.
"""
def plot_hours_by_income(df: pd.DataFrame, save: bool = True) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(8, 6))
    order = sorted(df["income"].unique())
    sns.boxplot(data=df, x="income", y="hours_per_week", order=order, hue="income", palette="Set2", legend=False, whis=3, ax=ax)
    ax.set_title("근무시간별 소득 분포 (Hours per Week by Income)")
    ax.set_xlabel("소득 구간 (Income)")
    ax.set_ylabel("주당 근무시간 (Hours per Week)")

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUT_DIR / "chart4_hours_by_income.png", dpi=150, bbox_inches="tight")
        print(f"[plot_hours_by_income] 저장됨: {OUTPUT_DIR / 'chart4_hours_by_income.png'}")

    return ax


if __name__ == "__main__":
    from src.data_loader import clean_pandas, load_pandas

    pdf = clean_pandas(load_pandas())
    plot_age_by_income(pdf)
    plot_correlation_heatmap(pdf)
    plot_hours_by_income(pdf)
