# 안양시 음식점 점포당 연매출 추정

경진대회 최종 모델은 랜덤 포레스트 회귀입니다. 밀당 지도 화면에 붙이려면 [프론트엔드 패치와 실행 방법](integration/README.md)을 참고하세요.

## 남긴 파일

| 파일 | 남긴 이유 |
|---|---|
| [`src/serve_randomforest.py`](src/serve_randomforest.py) | 기존 지도 화면의 분석 요청을 받아 저장된 모델 예측값을 반환합니다. |
| [`integration/frontend.patch`](integration/frontend.patch), [연동 가이드](integration/README.md) | 팀원의 완성된 웹앱에 분석 API를 연결하고 실행하는 데 필요합니다. |
| [`results/randomforest_model.pkl`](results/randomforest_model.pkl) | 학습된 전처리와 랜덤 포레스트가 들어 있습니다. 서비스에서 재학습 없이 사용합니다. |
| [`annual_per_store_training.csv`](data/processed/training/annual_per_store_training.csv), [입력 명세](data/processed/training/annual_training_manifest.json) | 서버가 인근 관측 상권을 찾고 모델의 9개 입력을 구성합니다. 재학습에서도 같은 데이터를 사용합니다. |
| [`src/train_single_model.py`](src/train_single_model.py), [`requirements-training.txt`](requirements-training.txt) | 기존 전처리와 80:20 분할로 모델을 다시 학습하는 데 필요합니다. |
| [`results/randomforest_metrics.csv`](results/randomforest_metrics.csv) | 팀원이 기존 학습·테스트 성능을 확인하고 재학습과 비교할 기준입니다. |
| `.gitignore`, 이 README | 로컬 환경 파일과 캐시를 제외하고, 실행과 정리 기준을 기록합니다. |

## 재학습

Python 3.13에서 저장소 루트를 기준으로 실행합니다. 예시 명령은 기존 모델과 평가 파일을 보존하도록 새 폴더에 결과를 씁니다.

```bash
python -m pip install -r requirements-training.txt
python src/train_single_model.py --results-dir work/retrained
```

2023·2024년 최종 입력 682행을 `random_state=42`로 섞어 학습 545행과 테스트 137행으로 나눕니다. 수치형 중앙값 대체와 범주형 결측치 처리 및 원핫 인코딩은 학습 세트에서만 계산합니다. 기존 테스트 R²는 0.1619, 평균 절대 오차는 약 2억 4,003만 원입니다. 같은 상권의 행이 학습·테스트 양쪽에 있을 수 있습니다.

학습 목표는 상권·업종별 4개 분기 추정매출 합계를 해당 연도의 관측 점포 수로 나눈 **점포당 연매출 참고값**입니다. 입력은 산업분류코드, 위도·경도, 대표점 반경 500m 경쟁 특성 3개, 행정동 유동인구 특성 3개입니다. 서버가 화면에 반환하는 월매출은 2024년 인근 동일 업종 대표점의 예측 연매출을 12로 나눈 값입니다.

## 제거한 파일

| 이전 파일 | 원래 역할과 제거 이유 |
|---|---|
| `data/raw/` | 매출, 상가업소, 상권현황, 유동인구, 임대료 원본 535개입니다. 서비스와 재학습은 이미 생성된 최종 CSV에서 시작합니다. 임대료는 최종 모델 입력에도 포함되지 않습니다. |
| `data/processed/`의 파일럿·중간 CSV와 검산표 | 분기별 매출과 점포 수 연결, 연매출 참고값 생성, 행정동 유동인구 집계, 지역·기간·목표값 검산에 쓰였습니다. 최종 CSV에 서버와 재학습에 필요한 값이 들어 있습니다. |
| `build_training_data.py`, `build_annual_training_source.py`, `prepare_model_data.py`, `precess_population.py` | 원본에서 분기·연간 학습 자료와 유동인구 입력을 만들던 코드입니다. 원본부터 최종 CSV를 다시 만들 때 사용합니다. 현재 브랜치의 재학습은 최종 CSV부터 실행합니다. |
| `build_anyang_sales.py`, `build_market_pilot.py`, `build_sales_training_pilot.py`, `match_sales_markets.py`, `prepare_anyang_sales.py` | 초기 안양 상권 매칭과 소규모 매출 파일럿 코드입니다. 현재 서버는 확정된 연간 데이터와 모델을 읽습니다. |
| `check_*.py`, `final_population_check.py`, `validata_population.py`, `test_population_duplicates.py`, `results/01_…`~`04_…` | 원본 파일의 구조·월별 누락·행정동 코드·중복과 상권 연결을 점검하던 코드 및 보고서입니다. 서비스 요청이나 모델 재학습 과정에서 사용하지 않습니다. |
| `randomforest_predictions.csv`, `randomforest_feature_importance.csv`·`.png` | 테스트 행별 예측과 변수 중요도 표·그래프입니다. 필요하면 위 재학습 명령으로 다시 생성합니다. |
| 루트·데이터·결과 폴더의 `.gitattributes` | 원본 CSV 및 생성물에 Git LFS와 예외 규칙을 적용했습니다. 현재 남긴 CSV는 일반 Git 파일로 관리합니다. |

원본부터 전처리를 다시 수행해야 하면 [정리 직전 커밋](https://github.com/kimwoojae1212/AI_project/tree/d3425a17057d7afaf838fc759e80f8697e882585)을 사용하세요. 파일은 현재 브랜치에서만 제거했으며 Git 이력은 유지했습니다.
