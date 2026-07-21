"""Seaborn 정적 차트 유틸리티.

data/processed/adult_clean.parquet (또는 src.data_loader 결과)을 입력으로 받아
차트를 그리고 output/ 에 저장한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import gaussian_kde

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 소득 카테고리 색상 = seaborn "Set2" 팔레트의 앞 두 색.
# 정적(Seaborn) 차트와 인터랙티브(Plotly) 차트가 동일한 그룹을
# 항상 같은 색으로 표시하도록 맞춘다.
INCOME_COLORS = {"<=50K": "#66c2a5", ">50K": "#fc8d62"}


def _mpl_colorscale(name: str, n: int = 21) -> list:
    """matplotlib 컬러맵을 Plotly color_continuous_scale 형식으로 변환.

    상관관계 히트맵에서 정적 차트와 동일한 "coolwarm" 그라데이션을
    쓰기 위함 (Plotly 내장 RdBu는 색 배치가 미묘하게 다르다).
    """
    cmap = plt.get_cmap(name)
    return [
        [frac, f"rgb({r * 255:.0f},{g * 255:.0f},{b * 255:.0f})"]
        for frac, (r, g, b, _a) in zip(np.linspace(0, 1, n), cmap(np.linspace(0, 1, n)))
    ]


COOLWARM = _mpl_colorscale("coolwarm")


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
    n = len(corr)
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, mask=mask_upper, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white", ax=ax,
    )

    # 위쪽 삼각형은 대칭이므로 숫자 대신 상관계수 크기에 비례한 원으로 표시
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1, n):
            value = corr.iloc[i, j]
            ax.scatter(
                j + 0.5, i + 0.5,
                s=abs(value) * 1800,
                color=cmap(norm(value)),
                edgecolor="white", linewidth=0.5,
            )

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


"""
나이(age) 분포를 소득(income) 그룹별로 비교하는 인터랙티브 히스토그램.

plot_age_by_income()의 Plotly 버전. histplot(kde=True)와 동일하게
히스토그램 위에 KDE 곡선을 겹쳐 그리고, 확대/축소·범례로 그룹
켜고 끄기·구간별 정확한 값 hover 확인이 가능하다.
"""
def plot_age_by_income_interactive(df: pd.DataFrame, save: bool = True) -> go.Figure:
    order = sorted(df["income"].unique())

    # sns.histplot(bins="auto")와 동일한 bin 경계를 재현.
    # numpy의 auto 규칙(Freedman-Diaconis/Sturges 중 더 촘촘한 쪽)이
    # 골라주는 폭(~1.2년)이 nbins로 대충 지정한 값보다 훨씬 촘촘해서,
    # 나이가 5년 단위로 몰리는(age-heaping) 뾰족한 막대 모양까지 살아난다.
    edges = np.histogram_bin_edges(df["age"], bins="auto")
    binwidth = edges[1] - edges[0]

    fig = px.histogram(
        df, x="age", color="income",
        category_orders={"income": order}, color_discrete_map=INCOME_COLORS,
        histnorm="probability density", barmode="overlay", opacity=0.5,
        title="나이 분포 (소득별) (Age Distribution by Income)",
        labels={"age": "나이 (Age)", "income": "소득"},
    )
    fig.update_traces(
        xbins=dict(start=edges[0], end=edges[-1], size=binwidth),
        selector=dict(type="histogram"),
    )

    # sns.histplot(kde=True)와 동일하게 그룹별 KDE 곡선을 겹쳐 그린다.
    x_grid = np.linspace(df["age"].min(), df["age"].max(), 200)
    for group in order:
        ages = df.loc[df["income"] == group, "age"]
        density = gaussian_kde(ages)(x_grid)
        fig.add_trace(go.Scatter(
            x=x_grid, y=density, mode="lines", name=f"{group} (KDE)",
            line=dict(color=INCOME_COLORS[group], width=2.5),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        yaxis_title="밀도 (Density)", legend_title_text="소득",
        template="plotly_white",
    )

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / "chart2_age_by_income_interactive.html"
        fig.write_html(out)
        print(f"[plot_age_by_income_interactive] 저장됨: {out}")

    return fig


"""
수치형 변수 간 상관관계 인터랙티브 Heatmap.

plot_correlation_heatmap()의 Plotly 버전. 아래쪽 삼각형은 정적
차트와 동일한 coolwarm 색상 + 값 표기이고, 위쪽 삼각형은 정적
차트의 "크기 원" 표현을 그대로 재현하되 마우스를 올리면 정확한
값도 함께 확인할 수 있다.
"""
def plot_correlation_heatmap_interactive(df: pd.DataFrame, save: bool = True) -> go.Figure:
    from src.data_loader import NUMERIC_COLUMNS

    corr = df[NUMERIC_COLUMNS].corr()
    cols = corr.columns.tolist()
    n = len(cols)
    lower = corr.mask(np.triu(np.ones((n, n), dtype=bool), k=1))

    fig = px.imshow(
        lower, text_auto=".2f", color_continuous_scale=COOLWARM,
        zmin=-1, zmax=1, color_continuous_midpoint=0, aspect="equal",
        title="수치형 변수 간 상관관계 (Correlation Heatmap)",
    )
    fig.update_traces(
        xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>r=%{z:.3f}<extra></extra>",
    )

    # 위쪽 삼각형: 정적 차트처럼 상관계수 크기에 비례한 원으로 표시.
    # 정적 차트는 matplotlib scatter의 s(면적, pt^2)=|r|*1800을 쓰므로,
    # Plotly의 size(지름, px)로 환산해 같은 시각적 크기가 되도록 맞춘다.
    upper_i, upper_j = np.triu_indices(n, k=1)
    values = corr.values[upper_i, upper_j]
    diameters = 2 * np.sqrt(np.abs(values) * 1800 / np.pi)
    fig.add_trace(go.Scatter(
        x=[cols[j] for j in upper_j], y=[cols[i] for i in upper_i],
        mode="markers",
        marker=dict(
            size=diameters,
            color=values, colorscale=COOLWARM, cmin=-1, cmax=1,
            line=dict(color="white", width=1), showscale=False,
        ),
        customdata=values,
        hovertemplate="%{y} · %{x}<br>r=%{customdata:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(template="plotly_white", coloraxis_colorbar_title="r")

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / "chart_correlation_heatmap_interactive.html"
        fig.write_html(out)
        print(f"[plot_correlation_heatmap_interactive] 저장됨: {out}")

    return fig


"""
근무시간(hours_per_week) vs 소득(income) 인터랙티브 Box plot.

plot_hours_by_income()의 Plotly 버전. 이상치 포인트에 마우스를
올리면 개별 값을 확인할 수 있다.
"""
def plot_hours_by_income_interactive(df: pd.DataFrame, save: bool = True) -> go.Figure:
    order = sorted(df["income"].unique())
    fig = px.box(
        df, x="income", y="hours_per_week", color="income",
        category_orders={"income": order}, color_discrete_map=INCOME_COLORS,
        points="outliers",
        title="근무시간별 소득 분포 (Hours per Week by Income)",
        labels={
            "income": "소득 구간 (Income)",
            "hours_per_week": "주당 근무시간 (Hours per Week)",
        },
    )
    # Plotly Express box는 marker.color를 박스 채우기 색으로도 쓰므로,
    # 먼저 채우기색을 고정해 둔 뒤 마커만 matplotlib 기본 이상치 스타일
    # (속이 빈 검은 원)로 바꾼다.
    for trace in fig.data:
        trace.fillcolor = trace.marker.color
        trace.marker.update(symbol="circle-open", color="black", size=6)
    fig.update_layout(showlegend=False, template="plotly_white")

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / "chart4_hours_by_income_interactive.html"
        fig.write_html(out)
        print(f"[plot_hours_by_income_interactive] 저장됨: {out}")

    return fig


if __name__ == "__main__":
    from src.data_loader import clean_pandas, load_pandas

    pdf = clean_pandas(load_pandas())
    plot_age_by_income(pdf)
    plot_correlation_heatmap(pdf)
    plot_hours_by_income(pdf)

    plot_age_by_income_interactive(pdf)
    plot_correlation_heatmap_interactive(pdf)
    plot_hours_by_income_interactive(pdf)
