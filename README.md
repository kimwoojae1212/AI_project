# 안양시 음식점 매출 모델 학습 데이터

## 팀 공통 연매출 학습 파일

모델 팀은 **[`annual_per_store_training.csv`](data/processed/training/annual_per_store_training.csv)** 와 [입력·목표 컬럼 명세](data/processed/training/annual_training_manifest.json)를 사용한다. 한 행은 **연도 × 상권 × 음식점 업종**이다. 안양 상권 67곳, 업종 11개, 총 682행이다.

| 역할 | 컬럼 |
|---|---|
| 정답 `y` | `점포당_연매출_참고값` (원) |
| 범주형 입력 | `산업분류코드` |
| 위치 입력 | 위도, 경도 |
| 경쟁 환경 입력 | 반경 500m 전체 음식점 수, 동종 음식점 수, 가장 가까운 동종 점포 거리(최대 500m) |
| 행정동 유동인구 입력 | 시간대 평균, 저녁(17~21시) 시간대 평균, 20~39세 비율 |
| 행 식별·분할 | 기준연도, 상권ID, 산업분류코드, `split` |

2023년 346행은 학습, 2024년 336행은 테스트다. 모델과 설정 비교는 **2023년 데이터에서 상권ID를 그룹으로 잡아 5-fold 교차검증**하고, 2024년은 선정된 모델의 최종 평가에 사용한다. CSV를 무작위 80:20 분할하면 같은 상권의 다른 행이 학습과 평가에 섞인다. `상권ID`, `기준연도`, `split`은 식별·검증용이다. 입력은 명세의 `feature_columns` 9개를 사용한다. 매출 원본과 분모는 별도 검산 파일에 기록했다.

### 전처리 및 검산

1. 같은 상권·업종에 **매출이 4개 분기 모두 있는 행**을 남겼다. 2023년 3월, 2024년 9월의 점포 파일에서 상권 폴리곤 안의 동종 점포를 센다. 연간 상권·업종 추정매출 합계를 관측 점포 수로 나눈 값을 목표로 사용한다. 원본 합계와 분모를 조인해 682행의 목표값을 검산했다.
2. 상권 대표점 주변의 500m 경쟁 특성을 만든다. 팀원 저장소의 2026년 10월 안양시 행정동 경계로 대표점을 공간 조인한다. 학습에서는 2023년 3월과 2024년 9월의 관측치를 사용한다.
3. 유동인구의 내·외국인 및 성별·연령대 집계를 합산해 행정동·날짜·시간별 인구를 만든다. 월별 평균과 연령대 비율은 관측된 시간 기준이다. 2023년 3월 누락 하루는 보간하지 않았다.
4. 안양시 조례에 따른 7개 행정동 명칭 변경을 연결했다. 예를 들어 과거 관양2동 자료를 현재 인덕원동 코드에 연결했다. 최초에는 153행에서 유동인구가 빠졌으며, 변경 전후 코드를 대조한 뒤 **682행 전체**가 연결됐다. 현재 경계 폴리곤을 과거 데이터에 적용하고 있으므로 과거 경계 형태의 차이까지 보정한 결과는 아니다.

[목표값과 매칭 검산표](data/processed/training/annual_training_audit.csv)에는 매출 합계와 관측 점포 수, 원본 행정동 코드, 관측 시간 수가 있다. [행정동 유동인구 집계표](data/processed/training/dong_population_features.csv)에는 향후 지도 입력에 붙이는 데 쓸 2025년 12월 값도 있다. [팀원 행정동 경계 원본](https://github.com/woochul0516/AI-project-univ/blob/7b0b9939b0b8be6ba5edce07e1d268138589664e/data/geojson/anyang-si.geojson) · [안양시 행정동 변동 이력](https://www.anyang.go.kr/main/contents.do?key=289) · [2024년 명칭 변경 공지](https://www.anyang.go.kr/main/selectBbsNttView.do?bbsNo=62&key=259&nttNo=350749)

### 모델 확인

아래 코드는 저장소 루트에서 실행한다. `requirements-training.txt`를 설치한 다음, 기존 학습 스크립트의 모델 부분을 비교 대상 모델로 바꿔도 입력과 분할은 동일하게 유지한다.

```python
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

root = Path("data/processed/training")
spec = json.loads((root / "annual_training_manifest.json").read_text(encoding="utf-8"))
columns, target = spec["feature_columns"], spec["target"]
data = pd.read_csv(root / spec["dataset"], dtype={"상권ID": str, "산업분류코드": str})
train = data.loc[data["split"].eq("train")]
test = data.loc[data["split"].eq("test")]

preprocess = ColumnTransformer([
    ("category", OneHotEncoder(handle_unknown="ignore"), spec["categorical_features"]),
    ("numeric", SimpleImputer(strategy="median"), spec["numeric_features"]),
])
model = make_pipeline(preprocess, RandomForestRegressor(n_estimators=200, random_state=42))
scores = cross_val_score(
    model, train[columns], train[target], groups=train["상권ID"],
    cv=GroupKFold(n_splits=5), scoring="neg_mean_absolute_error",
)
print(f"학습 연도 상권 그룹 CV MAE: {-scores.mean():,.0f}원")
model.fit(train[columns], train[target])
print(f"2024년 테스트 MAE: {mean_absolute_error(test[target], model.predict(test[columns])):,.0f}원")
medians = train.groupby("산업분류코드")[target].median()
baseline = test["산업분류코드"].map(medians).fillna(train[target].median())
print(f"업종별 중앙값 기준선 MAE: {mean_absolute_error(test[target], baseline):,.0f}원")
print(f"상권 수: {data['상권ID'].nunique()}, 업종 수: {data['산업분류코드'].nunique()}")
assert not data.duplicated(spec["identifier_columns"]).any()
assert not data[columns + [target]].isna().any().any()
```

랜덤 포레스트의 2024년 테스트 MAE는 **169,703,187원**이었다. 업종별 중앙값 기준선의 MAE는 **203,615,189원**이었다. 기존 경쟁 특성만 쓴 데모의 랜덤 포레스트는 **165,389,119원**이었으므로, 이번 실행에서 유동인구 추가로 테스트 MAE가 개선되지는 않았다. 동일 데이터로 팀 모델별 비교는 가능하며, 이 수치는 신규 점포의 실측 매출이나 첫 영업연도 예측 정확도를 의미하지 않는다.

### 다시 만들기

```bash
python -m pip install -r requirements-training.txt
git lfs pull --include="data/processed/안양시_유동인구_2023_2025.csv"
curl -fL "https://raw.githubusercontent.com/woochul0516/AI-project-univ/7b0b9939b0b8be6ba5edce07e1d268138589664e/data/geojson/anyang-si.geojson" -o /tmp/anyang-si.geojson
python src/prepare_model_data.py --boundary /tmp/anyang-si.geojson
```

전처리는 기존 분기 CSV 두 개와 연매출 참고값 원본을 읽는다. 이 파일들까지 원본에서 다시 만들 때는 아래의 `build_training_data.py`와 `predict_annual_sales.py build --proxy`를 순서대로 실행한다. 기존 `/predict` 데모는 경쟁 특성만 사용한다. 새 모델을 붙일 때는 같은 지도 경계로 입력점을 행정동에 연결하고, 사용 가능한 기준월의 유동인구 집계표를 붙여 명세의 9개 열을 전달한다.

**정답값의 한계**는 연도별로 한 시점의 점포 수를 사용한다는 점이다. 4개 분기 말 점포 수가 확보되면 분기마다 같은 분기의 점포 수를 적용할 수 있다. 신규 가게 첫 12개월 실매출을 검증하려면 개업 시점과 점포 단위 매출 자료가 필요하다. 임대료는 이번 학습 파일에 포함하지 않았다.

## 지도에서 음식점의 연매출을 예측하는 데모

지도에서 클릭한 좌표와 표준산업분류코드를 입력하면, **반경 500m 전체 음식점 수, 동종 음식점 수, 가장 가까운 동종 음식점 거리**를 계산해 가게 한 곳의 예상 연매출 참고값을 반환한다. 거리는 직선거리다. 반경에 동종점이 없으면 점포 수는 0이고 거리 특성은 500m로 기록된다.

아래 기존 API 데모는 유동인구를 붙이기 전의 `data/processed/training/annual_per_store_proxy.csv`를 사용한다. 한 행은 `연도 × 상권ID × 업종코드`이고, 목표값은 `점포당_연매출_참고값`이다. 미래 분기를 학습에 섞지 않도록 2023년 346행을 학습, 2024년 336행을 테스트로 지정했다. 입력 특성은 `산업분류코드`, `위도`, `경도`, `500m_음식점수`, `500m_동종점포수`, `동종점포_최근거리_m_상한500`이다.

### 실행

```bash
python -m pip install -r requirements-training.txt
python src/predict_annual_sales.py evaluate
git lfs pull --include="data/raw/상권/**"
python src/predict_annual_sales.py predict --lat 37.38557 --lon 126.93186 --code 56122
python src/predict_annual_sales.py serve --port 8000
```

서버 실행 후 다른 터미널에서 호출할 수 있다.

```bash
curl "http://127.0.0.1:8000/predict?lat=37.38557&lon=126.93186&code=56122"
```

프론트엔드는 지도 클릭 좌표와 업종 선택값을 위 요청에 넘긴다. 응답에 매출(원), 사용한 점포 자료의 기준연월, 주변 음식점 지표가 포함된다. 학습에 없던 업종이거나 가장 가까운 동종 업종 학습 상권 대표점에서 1km를 넘으면 HTTP 400과 사유를 반환한다. 1km는 데모에서 정한 지원 기준이다. 이 API는 로컬용이며 지도 프론트엔드 연결과 배포는 별도로 진행한다.

### 목표값과 데이터 범위

- 데모용 정답은 **실제 네 분기의 상권·업종별 추정매출 합계 ÷ 해당 연도에 확보한 한 시점의 관측 점포 수**다. 2023년 3월, 2024년 9월을 사용한다. 실제 가게별 연매출과 차이가 날 수 있고, 연중 개폐업 규모도 반영하지 못한다. 2025년은 매출 네 분기가 없어서 학습에 넣지 않았다.
- 훈련에서는 상권 대표점과 해당 연도의 점포 스냅샷으로 주변 변수를 계산한다. 예측에서는 사용자가 클릭한 점과 최신 스냅샷을 쓴다. 연도 비교는 점포당 연매출 **참고값**의 오차를 보여준다. 신규 상권이나 개별 신규 점포의 실매출 예측 성능은 별도 검증이 필요하다.
- 현재 주변 경쟁 점포 정보와 위치·업종을 사용한다. 유동인구, 임대료, 점포 면적은 반영하지 않았다. 예측 지역도 공식 안양시 경계 대신 학습 상권과의 거리로 확인한다.
- 분기별 스냅샷이 확보되면 `build`를 `--proxy` 없이 실행한다. 각 분기의 매출을 같은 분기 점포 수로 나누고, 모두 연결된 상권·업종의 4분기 값을 합한다. 평가 및 예측 시 `--dataset data/processed/training/annual_per_store.csv`를 하위 명령 앞에 지정한다.

```bash
git lfs pull --include="data/raw/매출/**,data/raw/상권/**,data/raw/상권현황/**"
python src/build_training_data.py
python src/predict_annual_sales.py build --proxy
```

## 분기 매출 데이터와 기존 모델 비교

아래는 분기 매출을 비교하는 기존 학습 데이터의 사용법이다. 지도에서 한 점포의 연매출을 예측하는 데모는 위의 연매출 CSV를 사용한다.

`data/processed/training/`에서 전처리한 CSV를 바로 읽을 수 있다. 세 모델 모두 같은 파일과 `split`을 사용한다. Python 3.10 이상에서 저장소 루트 기준으로 실행한다.

## 어떤 CSV를 학습하나요?

| 파일 | 한 행의 단위 | 목표값 | 용도 |
|---|---|---|---|
| `quarterly_market_sales.csv` | 상권 × 음식점 업종 × 분기 | `분기_상권업종_추정매출`(원) | 여러 분기의 데이터로 팀 공통 모델 비교 |
| `quarterly_per_store_sales.csv` | 상권 × 음식점 업종 × 분기 | `점포당_분기매출_참고값`(원) | 매출과 점포 스냅샷이 같은 분기인 경우의 보조 실험 |

첫 번째 CSV의 목표값은 **상권 안의 해당 업종 전체** 추정매출이다. 두 번째는 상권·업종의 추정매출을 분기 말에 관측한 해당 업종 점포 수로 나눈 값이다. 두 결과의 단위가 다르므로 모델 점수를 서로 비교할 수 없다. 두 CSV 모두 개별 가게의 연매출 실측값은 제공하지 않는다.

`dataset_manifest.json`에 입력 변수와 목표값, 분할별 행 수가 들어 있다. 입력 변수는 상권 ID, 산업분류코드, 분기, 상권 대표 좌표다. `분기말_관측점포수`는 점포당 목표값의 산식 확인용으로 남겨뒀다. 개업 전 예측 시점에 분기 말 점포 수를 알 수 없으므로 기본 입력에는 넣지 않는다. 업종은 원본의 `561xx` 범위를 사용한다.

현재 원본에서 상권 69개를 포함했다. 검증·테스트 기간에는 동일한 분기의 행 전체가 함께 들어간다.

| 데이터 | 학습 | 검증 | 테스트 | 전체 |
|---|---:|---:|---:|---:|
| 상권·업종 총매출 | 4,693 (2023년 1분기~2024년 3분기) | 685 (2024년 4분기) | 682 (2025년 3분기) | 6,060 |
| 점포당 참고값 | 353 (2023년 1분기) | — | 344 (2024년 3분기) | 697 |

점포당 파일은 분기가 두 개이므로 테스트셋을 반복해서 보며 모델 설정을 선택하지 않는다.

## 바로 학습해 보기

```bash
python -m pip install -r requirements-training.txt
```

아래 예시는 그대로 실행할 수 있다. 랜덤 포레스트를 팀원별 모델로 바꾸더라도 입력 열과 분할은 그대로 사용한다. 검증셋은 설정 비교에, 테스트셋은 최종 평가에 사용한다.

```python
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

root = Path("data/processed/training")
name = "quarterly_market_sales.csv"
spec = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["datasets"][name]
categories = ["상권ID", "산업분류코드", "기준분기"]
data = pd.read_csv(root / name, dtype={column: str for column in categories})
features, target = spec["features"], spec["target"]

encode = ColumnTransformer([
    ("category", OneHotEncoder(handle_unknown="ignore"), categories),
    ("coordinates", "passthrough", ["경도", "위도"]),
])
model = make_pipeline(encode, RandomForestRegressor(n_estimators=100, random_state=42))
train = data.loc[data["split"].eq("train")]
model.fit(train[features], train[target])

for name in ("validation", "test"):
    fold = data.loc[data["split"].eq(name)]
    if fold.empty:
        continue
    predicted = model.predict(fold[features])
    print(f"{name} MAE: {mean_absolute_error(fold[target], predicted):,.0f}원")
```

## 원본에서 다시 생성하기

원본은 Git LFS로 저장돼 있다. Git LFS가 설치된 환경에서 다음 명령으로 필요한 원본만 받고 생성한다.

```bash
git lfs pull --include="data/raw/매출/**,data/raw/상권/**,data/raw/상권현황/**"
python src/build_training_data.py
```

| 단계 | 처리 |
|---|---|
| 매출 정제 | 연도별 원본에서 음식점 업종을 통합하고, 공백 및 중복을 정리한다. 같은 분기·상권·업종에 서로 다른 매출이 있으면 중단한다. |
| 상권 연결 | 매출의 상권ID·이름과 상권현황의 이름이 1대1로 대응하는 유효한 폴리곤을 찾는다. 기존에 확인한 두 상권을 포함한다. 추가 상권은 최초 스냅샷에서 폴리곤 안에 안양시 점포가 1개 이상 있고 타 시·군 점포가 관측되지 않는 경우 포함한다. |
| 점포 수 집계 | 분기 말 스냅샷이 있는 분기에만 안양시 점포 좌표와 상권 폴리곤을 연결한다. 동일 상권·업종의 관측 점포가 있는 행에 점포당 참고값을 계산한다. |
| 시간순 분할 | 가장 최근 분기가 테스트, 두 번째로 최근 분기가 검증이다. 전체 기간이 두 개면 이전 분기가 학습, 이후 분기가 테스트다. |

`market_audit.csv`는 이름으로 연결한 경계 후보의 선정 근거와 기준 점포 수를 보여준다. `period_audit.csv`는 실제 매출 기간, 사용 가능한 분기 말 스냅샷, 두 데이터의 유효 행 수를 보여준다.

## 범위와 제한

- 2025년 매출 원본의 `기준분기`는 3분기뿐이다. 2025년 12월 점포 스냅샷은 4분기에 해당하므로 3분기 매출의 분모로 사용하지 않았다.
- 추가 상권은 점포 위치와 시군구 표기로 추정했다. 공식 안양시 경계로 검증한 전수 목록이 아니며, 처음 보는 상권에 대한 성능은 이 시간 분할로 판단할 수 없다.
- 유동인구와 임대료는 기본 모델에서 제외했다. 결합하려면 검증된 상권–행정동 매핑과 임대료의 지역·시점 단위를 확보하고, 서비스에서 예측 시점에 알 수 있는 값만 사용해야 한다.
- 최종 서비스에 점포당 예상 연매출을 표시하려면 분기별 점포당 목표값을 만들 수 있는 원본을 더 확보하고, 여러 분기를 합산한 결과를 별도로 검증해야 한다.
