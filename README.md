# SKALA Capstone2 — Day2 종합 실습 (Adult Census Income)

End2End 데이터 분석 프로젝트. 데이터셋: **Adult Census Income** (UCI).

> **⚠️ 데이터는 git에 포함되어 있지 않습니다.** `data/` 안의 실제 파일들은
> `.gitignore`로 제외되어 있고 `.gitkeep`만 커밋됩니다. clone 직후에는
> `data/raw/`, `data/processed/` 가 비어 있으니 아래 "팀원 온보딩" 순서를
> 반드시 먼저 실행하세요. **zip 제출 전에도 이 스크립트를 실행한 상태여야
> 데이터가 함께 담깁니다.**

## 팀원 온보딩 (처음 clone 했을 때 한 번)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh          # data/raw/ 채우기
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# → data/processed/adult_clean.parquet 생성됨. 이후 시각화·통계·ML 담당자는
#   이 parquet 또는 src.data_loader의 load_pandas()/load_polars()를 사용
```

## 프로젝트 구조

```
.
├── data/
│   ├── raw/            # 원본 데이터 (git 미포함, download_data.sh로 재현)
│   └── processed/      # 전처리 완료 데이터
├── notebooks/          # EDA·실험용 Jupyter 노트북
├── src/                # 재사용 가능한 Python 모듈
│   └── data_loader.py  # Pandas/Polars 로딩 + 클린 + 기본 EDA
├── scripts/
│   └── download_data.sh
├── tests/               # pytest
├── output/               # 리포트/차트 산출물
├── requirements.txt
├── pyproject.toml       # ruff / pytest 설정
└── .venv/               # 가상환경 (git 미포함)
```

## 환경 설정

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

```bash
bash scripts/download_data.sh   # data/raw/adult.data, adult.names, adult.test 다운로드
jupyter lab notebooks/01_eda.ipynb   # Pandas/Polars 로딩 비교 + 결측치/중복 처리 + 기본 EDA
```

로딩·클린 함수는 `src/data_loader.py` 에 있고, 실제 탐색·실행은 `notebooks/01_eda.ipynb` 에서 한다
(`load_pandas()` / `load_polars()` / `clean_pandas()` / `basic_eda()`).

컬럼: `age, workclass, fnlwgt, education, education_num, marital_status,
occupation, relationship, race, sex, capital_gain, capital_loss,
hours_per_week, native_country, income` (총 15개, 32,561행)

- 결측치는 원본에서 `"?"` 로 표기되어 있으며 로딩 시 NaN/null 로 변환된다.
- `income` 이 타깃 컬럼 (`<=50K` / `>50K`).

## 테스트 · 린트

```bash
pytest tests/ -v
ruff check .
```

## 역할 분담 (Day2 종합 실습)

- **데이터 준비**: `src/data_loader.py`, `notebooks/01_eda.ipynb` — Pandas/Polars 로딩 비교, 결측치·중복 처리, 기본 EDA
- **시각화**: Seaborn 정적 차트 + Plotly 인터랙티브 차트
- **통계 분석**: 기술통계, 상관계수, t-test
- **ML Pipeline**: sklearn Pipeline + joblib 저장
- **자동화·발표**: report.md 자동 생성

## 제출 전 체크리스트

- [ ] `bash scripts/download_data.sh` 실행 완료 (`data/raw/` 에 파일 존재 확인)
- [ ] `notebooks/01_eda.ipynb` 실행 완료 (`data/processed/adult_clean.parquet` 존재 확인)
- [ ] `pytest tests/ -v`, `ruff check .` 통과
- [ ] GitHub에서 최신 폴더 구조 그대로 다운로드 후 zip (`캠퍼스명_반_이름_day2종합실습.zip`)
