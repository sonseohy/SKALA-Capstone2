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
│   ├── raw/                              # 다운로드한 Adult 원본 데이터
│   └── processed/                        # 결측치·중복 처리 후 데이터
├── notebooks/
│   ├── 01_eda.ipynb                     # 데이터 탐색과 기본 전처리
│   ├── 02_visualization.ipynb           # Seaborn 기반 정적 시각화
│   └── 03_visualization_interactive.ipynb # Plotly 기반 인터랙티브 시각화
├── src/
│   ├── data_loader.py                   # 데이터 로딩·정제와 기본 EDA
│   ├── visualize.py                     # 정적·인터랙티브 차트 함수
│   ├── statistical_analysis.py          # 기술통계와 가설검정
│   └── pipeline/
│       ├── model_preprocess.py          # 학습 전 전처리 파이프라인
│       └── model_train.py               # 모델 학습·평가와 저장
├── scripts/
│   └── download_data.sh                 # UCI 데이터 다운로드
├── tests/
│   ├── test_data_loader.py              # 데이터 처리 테스트
│   └── test_model_train.py              # 모델 파이프라인 테스트
├── output/
│   └── adult_income_pipeline.joblib     # 학습이 끝난 모델 파이프라인
├── ADULT_CODEBOOK.md                    # 데이터 변수 설명
├── statistical_analysis_report.md       # 통계 분석 결과
├── requirements.txt                     # 실행에 필요한 라이브러리
├── pyproject.toml                       # Ruff·pytest 설정
├── .gitignore                           # Git 제외 파일 설정
├── .vscode/settings.json                # VS Code 프로젝트 설정
└── .claude/settings.json                # 프로젝트 도구 권한 설정
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

- 결측치는 원본에서 `"?"` 로 표기되어 있으며 로딩 시 NaN/null 로 변환되고,
  `clean_pandas()` 단계에서 결측치가 있는 행은 제거된다 (32,561행 → 정제 후 30,139행).
- `income` 이 타깃 컬럼 (`<=50K` / `>50K`).

### 이상치 처리에 대해

`clean_pandas()` 는 결측치·중복행만 제거하고, **수치 컬럼에 대한 IQR 등 이상치 제거는
의도적으로 수행하지 않는다.** 컬럼별 극단값을 검토한 결과, 대부분 데이터 오류가
아니라 실제 인구 다양성이자 타깃(`income`) 예측에 유의미한 신호로 판단했기 때문이다.

- `capital_gain` / `capital_loss`: 대부분 0이라 IQR이 0으로 계산되어(0이 아닌 값이
  전부 이상치로 잡히는 식으로) 표준 IQR 방식 자체가 이 컬럼에 맞지 않는다. 0이 아닌
  값(고액 자본소득/손실)은 오히려 고소득 여부를 가르는 핵심 신호다.
- `hours_per_week`: 1.5×IQR 기준 26.4%, 3×IQR 기준에도 12.4%가 제거될 만큼 분포가
  넓다. 파트타임(학생 등)부터 장시간 근무(자영업 등)까지 실제 존재하는 정상 패턴이며
  income과 상관관계가 있다.
- `fnlwgt`: 사람의 속성이 아니라 센서스 표본 가중치(sampling weight)라서, 극단값은
  가중치 산출 방식에 따른 정상적인 결과이지 이상치가 아니다.
- `age`, `education_num`: 3×IQR 기준으로는 사실상 이상치가 거의 없다(각각 0%). age의
  상한(90)은 census 특유의 top-coding이고, education_num은 학력 단계를 그대로
  숫자화한 이산형 서열값이라 애초에 "이상치" 개념이 잘 맞지 않는다.

결론적으로 이 데이터셋은 극단값을 잘라내면 오히려 고소득/장시간근무 등 예측에
중요한 케이스를 편향적으로 제거하게 된다. 극단값을 다뤄야 하는 경우 이상치
"제거"보다는 모델링 단계에서 로버스트 스케일러(RobustScaler)나 트리 기반 모델
(RandomForest, GBM 등 스케일에 덜 민감한 모델) 사용을 권장한다.

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
