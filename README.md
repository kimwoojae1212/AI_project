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

### 경진대회 최종 모델 학습

경진대회 최종 모델은 랜덤 포레스트 회귀(`RandomForestRegressor`)다. 팀 모델 비교와 동일하게 2023·2024년 전체 682행을 섞어 `train_test_split(test_size=0.2, random_state=42)`로 나눈다. 학습 545행과 테스트 137행을 사용하며, 같은 상권이 양쪽에 포함될 수 있다. 위의 연도 분할은 2023년으로 학습한 모델의 2024년 성능을 평가한다. 두 방식은 평가 조건이 다르므로 같은 분할로 산출한 모델 지표끼리 비교한다.

```bash
python -m pip install -r requirements-training.txt
python src/train_single_model.py
```

학습 입력은 명세의 9개 컬럼이며 목표는 `점포당_연매출_참고값`이다. 수치형 중앙값과 범주형 원핫 인코딩은 학습 세트에서만 추정한다. `results/`에 학습·테스트 지표, 테스트 행별 예측과 절대오차, 원본 변수별 중요도 CSV 및 PNG, 전처리를 포함한 모델을 `randomforest_` 접두어로 저장한다. MAPE는 백분율로 기록하며 실제값이 0인 행은 MAPE 계산에서 제외한다. 학습 시간은 전체 파이프라인의 1회 학습 시간이고 추론 시간은 각 세트 전체의 예측 시간이다. 그래프에는 설치된 한글 글꼴을 사용하며, `--font-path`로 글꼴 파일을 지정할 수 있다. 저장된 모델은 Python 3.13과 scikit-learn 1.9.0에서 생성했다.

### 백엔드에서 예측하기

GitHub의 `training-data-pipeline` 브랜치에서 [학습된 모델](results/randomforest_model.pkl)과 [입력 명세](data/processed/training/annual_training_manifest.json)를 가져온다. Python 3.13 환경에서 아래 버전을 사용한다.

```bash
python -m pip install numpy==2.4.4 pandas==3.0.3 scikit-learn==1.9.0
```

아래 예시는 저장소 루트에서 실행한다. 백엔드에서는 모델과 명세를 서버 시작 시 한 번 읽고, 요청 처리 시 `predict_annual_sales(payload)`를 호출한다. 역직렬화에는 이 저장소에서 받은 모델 파일을 사용한다.

```python
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(".")
manifest_path = root / "data/processed/training/annual_training_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
features = manifest["feature_columns"]

with (root / "results/randomforest_model.pkl").open("rb") as model_file:
    model = pickle.load(model_file)


def predict_annual_sales(payload):
    row = {name: np.nan if payload[name] is None else payload[name] for name in features}
    inputs = pd.DataFrame([row], columns=features)
    return float(model.predict(inputs)[0])
```

요청에는 명세의 9개 키를 모두 넣는다. `산업분류코드`는 문자열, 나머지 8개 변수는 숫자로 전달한다. 관측값이 없으면 `null`을 전달한다. 모델에 저장된 전처리가 결측치 처리와 범주형 인코딩을 수행한다. API 응답의 예측값 단위는 원이며, 화면에는 **점포당 연매출 참고 추정치**로 표시한다.

### 전처리 및 검산

1. 같은 상권·업종에 **매출이 4개 분기 모두 있는 행**을 남겼다. 2023년 3월, 2024년 9월의 점포 파일에서 상권 폴리곤 안의 동종 점포를 센다. 연간 상권·업종 추정매출 합계를 관측 점포 수로 나눈 값을 목표로 사용한다. 원본 합계와 분모를 조인해 682행의 목표값을 검산했다.
2. 상권 대표점 주변의 500m 경쟁 특성을 만든다. 팀원 저장소의 2026년 10월 안양시 행정동 경계로 대표점을 공간 조인한다. 학습에서는 2023년 3월과 2024년 9월의 관측치를 사용한다.
3. 유동인구의 내·외국인 및 성별·연령대 집계를 합산해 행정동·날짜·시간별 인구를 만든다. 월별 평균과 연령대 비율은 관측된 시간 기준이다. 2023년 3월 누락 하루는 보간하지 않았다.
4. 안양시 조례에 따른 7개 행정동 명칭 변경을 연결했다. 예를 들어 과거 관양2동 자료를 현재 인덕원동 코드에 연결했다. 최초에는 153행에서 유동인구가 빠졌으며, 변경 전후 코드를 대조한 뒤 **682행 전체**가 연결됐다. 현재 경계 폴리곤을 과거 데이터에 적용하고 있으므로 과거 경계 형태의 차이까지 보정한 결과는 아니다.

[목표값과 매칭 검산표](data/processed/training/annual_training_audit.csv)에는 매출 합계와 관측 점포 수, 원본 행정동 코드, 관측 시간 수가 있다. [행정동 유동인구 집계표](data/processed/training/dong_population_features.csv)에는 향후 지도 입력에 붙이는 데 쓸 2025년 12월 값도 있다. [팀원 행정동 경계 원본](https://github.com/woochul0516/AI-project-univ/blob/7b0b9939b0b8be6ba5edce07e1d268138589664e/data/geojson/anyang-si.geojson) · [안양시 행정동 변동 이력](https://www.anyang.go.kr/main/contents.do?key=289) · [2024년 명칭 변경 공지](https://www.anyang.go.kr/main/selectBbsNttView.do?bbsNo=62&key=259&nttNo=350749)

### 다시 만들기

```bash
python -m pip install -r requirements-training.txt
git lfs pull --include="data/processed/안양시_유동인구_2023_2025.csv"
curl -fL "https://raw.githubusercontent.com/woochul0516/AI-project-univ/7b0b9939b0b8be6ba5edce07e1d268138589664e/data/geojson/anyang-si.geojson" -o /tmp/anyang-si.geojson
python src/prepare_model_data.py --boundary /tmp/anyang-si.geojson
```

전처리는 기존 분기 CSV 두 개와 연매출 참고값 원본을 읽는다. 이 중간 파일도 원본에서 다시 만들려면 다음 순서로 실행한다.

```bash
git lfs pull --include="data/raw/매출/**,data/raw/상권/**,data/raw/상권현황/**"
python src/build_training_data.py
python src/build_annual_training_source.py --proxy
python src/prepare_model_data.py --boundary /tmp/anyang-si.geojson
```

**정답값의 한계**는 연도별로 한 시점의 점포 수를 사용한다는 점이다. 4개 분기 말 점포 수가 확보되면 분기마다 같은 분기의 점포 수를 적용할 수 있다. 신규 가게 첫 12개월 실매출을 검증하려면 개업 시점과 점포 단위 매출 자료가 필요하다. 임대료는 이번 학습 파일에 포함하지 않았다.
