# 안양시 음식점 점포당 연매출 추정

경진대회 최종 모델은 랜덤 포레스트 회귀입니다. 밀당 지도 화면에 붙이려면 [프론트엔드 패치와 실행 방법](integration/README.md)을 참고하세요.

## 서비스 실행과 재학습에 쓰는 파일

| 파일 | 남긴 이유 |
|---|---|
| [`src/serve_randomforest.py`](src/serve_randomforest.py) | 기존 지도 화면의 분석 요청을 받아 저장된 모델 예측값을 반환합니다. |
| [`integration/frontend.patch`](integration/frontend.patch), [연동 가이드](integration/README.md) | 팀원의 완성된 웹앱에 분석 API를 연결하고 실행하는 데 필요합니다. |
| [`results/randomforest_model.pkl`](results/randomforest_model.pkl) | 학습된 전처리와 랜덤 포레스트가 들어 있습니다. 서비스에서 재학습 없이 사용합니다. |
| [`annual_per_store_training.csv`](data/processed/training/annual_per_store_training.csv), [입력 명세](data/processed/training/annual_training_manifest.json) | 서버가 인근 관측 상권을 찾고 모델의 9개 입력을 구성합니다. 재학습에서도 같은 데이터를 사용합니다. |
| [`src/train_single_model.py`](src/train_single_model.py), [`requirements-training.txt`](requirements-training.txt) | 기존 전처리와 80:20 분할로 모델을 다시 학습하는 데 필요합니다. |
| [`results/randomforest_metrics.csv`](results/randomforest_metrics.csv) | 팀원이 기존 학습·테스트 성능을 확인하고 재학습과 비교할 기준입니다. |
| [`data/raw/`](data/raw), `.gitattributes` | 매출·상가업소·상권현황·유동인구·임대료 원본 CSV 535개와 원본에 적용되는 Git LFS 규칙을 보관합니다. |

## 재학습

Python 3.13에서 저장소 루트를 기준으로 실행합니다. 예시 명령은 기존 모델과 평가 파일을 보존하도록 새 폴더에 결과를 씁니다.

```bash
python -m pip install -r requirements-training.txt
python src/train_single_model.py --results-dir work/retrained
```

2023·2024년 최종 입력 682행을 `random_state=42`로 섞어 학습 545행과 테스트 137행으로 나눕니다. 수치형 중앙값 대체와 범주형 결측치 처리 및 원핫 인코딩은 학습 세트에서만 계산합니다. 기존 테스트 R²는 0.1619, 평균 절대 오차는 약 2억 4,003만 원입니다. 같은 상권의 행이 학습·테스트 양쪽에 있을 수 있습니다.

학습 목표는 상권·업종별 4개 분기 추정매출 합계를 해당 연도의 관측 점포 수로 나눈 **점포당 연매출 참고값**입니다. 입력은 산업분류코드, 위도·경도, 대표점 반경 500m 경쟁 특성 3개, 행정동 유동인구 특성 3개입니다. 서버가 화면에 반환하는 월매출은 2024년 인근 동일 업종 대표점의 예측 연매출을 12로 나눈 값입니다.

로컬 환경 파일과 캐시는 `.gitignore`에서 제외합니다.

## 발표에서 확인할 과정과 결과

1. **원본 수집과 상권 매칭.** [원본 CSV](data/raw/)에서 안양시 상권과 매출을 연결합니다. [상권 매칭 검산표](data/processed/training/market_audit.csv)에 상권을 포함하거나 제외한 근거를 남겼고, [분기 집계 검산표](data/processed/training/period_audit.csv)에 분기별 점포 관측 여부와 매출 행 수를 기록했습니다. [분기별 상권 매출](data/processed/training/quarterly_market_sales.csv)과 [분기별 점포당 매출](data/processed/training/quarterly_per_store_sales.csv)은 이 단계의 산출물입니다.
2. **목표값과 입력 변수 생성.** 매출이 4분기 모두 있는 상권·업종을 남기고, 연간 추정매출 합계를 해당 연도에 관측한 동종 점포 수로 나눴습니다. [연매출 중간 자료](data/processed/training/annual_per_store_proxy.csv)에서 500m 경쟁 변수까지 확인할 수 있고, [연매출 검산표](data/processed/training/annual_training_audit.csv)에는 분자·분모와 매칭 근거가 들어 있습니다.
3. **행정동 유동인구 결합.** [안양시 유동인구 통합 CSV](data/processed/안양시_유동인구_2023_2025.csv)에서 [월별 행정동 특성](data/processed/training/dong_population_features.csv)을 집계했습니다. 2023년 3월의 누락 하루는 보간하지 않았습니다. 변경 전후 행정동 코드를 대조해 초기 미매칭 153행을 해소했고 **최종 682행 모두 연결**했습니다. 행정동은 현재 경계에 맞춰 연결했으므로 과거 경계 모양까지 재현한 자료는 아닙니다.
4. **학습과 결과 해석.** 팀의 모델 비교 조건에 맞춰 80:20으로 나눴습니다. [학습·테스트 성능표](results/randomforest_metrics.csv), [테스트 137행의 실제값·예측값·오차](results/randomforest_predictions.csv), [변수 중요도 표](results/randomforest_feature_importance.csv)와 [발표용 그래프](results/randomforest_feature_importance.png)를 확인할 수 있습니다. 중요도는 모델이 분할에 사용한 정도이며 변수의 인과 효과로 해석할 수 없습니다. 원본의 연도별 `split` 컬럼은 별도 평가용 구분이고, 이 랜덤 포레스트 결과는 무작위 80:20 분할에서 나온 값입니다.
5. **서비스 연결.** [프론트엔드 패치와 API 실행 가이드](integration/README.md)에서 저장된 모델을 지도 화면에 연결하는 방법을 확인할 수 있습니다.

### 전처리 재현 순서

실행 위치는 저장소 루트입니다. 아래 순서는 기본 출력 경로의 중간 자료와 최종 학습 CSV를 다시 생성하므로 별도 체크아웃에서 실행하는 편이 좋습니다. 기존 [팀원 행정동 경계 파일](https://github.com/woochul0516/AI-project-univ/blob/7b0b9939b0b8be6ba5edce07e1d268138589664e/data/geojson/anyang-si.geojson)을 준비합니다.

```bash
python -m pip install -r requirements-training.txt
git lfs pull --include="data/raw/매출/**,data/raw/상권/**,data/raw/상권현황/**,data/processed/안양시_유동인구_2023_2025.csv"
python src/build_training_data.py
python src/build_annual_training_source.py --proxy
python src/prepare_model_data.py --boundary path/to/anyang-si.geojson
```

유동인구 통합 CSV를 원본 36개에서 다시 만들 때는 해당 원본 LFS 자료를 받은 후 `python src/precess_population.py`를 실행합니다. `build_sales_training_pilot.py`는 이름에 파일럿이 들어가지만 분기 매출 코드가 상권 기준과 데이터 키를 이 파일에서 가져오므로 유지합니다.

## 이번 정리에서 뺀 자료

현재 브랜치에서 제거 상태로 둔 것은 초기 두 상권 확인용 샘플, 소규모 매출 파일럿 산출물, 반복 실행용 파일 목록·컬럼 검사 코드와 탐색 보고서입니다. 이런 자료의 역할은 초기 연결 가능성과 파일 구조를 확인하는 것이었습니다. 최종 단계의 상권 선정 근거와 분기·연매출·인구 검산은 위 표와 코드에서 확인할 수 있습니다. 탐색 자료가 필요하면 [정리 전 Git 이력](https://github.com/kimwoojae1212/AI_project/tree/d3425a17057d7afaf838fc759e80f8697e882585)에서 볼 수 있습니다.
