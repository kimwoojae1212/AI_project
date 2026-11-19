# 안양시 음식점 매출 모델 학습 데이터

## 지도에서 음식점의 연매출을 예측하는 데모

지도에서 클릭한 좌표와 표준산업분류코드를 입력하면, **반경 500m 전체 음식점 수, 동종 음식점 수, 가장 가까운 동종 음식점 거리**를 계산해 가게 한 곳의 예상 연매출 참고값을 반환한다. 거리는 직선거리다. 반경에 동종점이 없으면 점포 수는 0이고 거리 특성은 500m로 기록된다.

팀 공통 연매출 데이터는 `data/processed/training/annual_per_store_proxy.csv`다. 한 행은 `연도 × 상권ID × 업종코드`이고, 목표값은 `점포당_연매출_참고값`이다. 미래 분기를 학습에 섞지 않도록 2023년 346행을 학습, 2024년 336행을 테스트로 지정했다. 입력 특성은 `산업분류코드`, `위도`, `경도`, `500m_음식점수`, `500m_동종점포수`, `동종점포_최근거리_m_상한500`이다.

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
