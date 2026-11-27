# 밀당 웹앱에 랜덤 포레스트 연결

팀 웹앱 [AI-project-univ의 기준 커밋](https://github.com/woochul0516/AI-project-univ/tree/97c37f4cccd5f7d78e3225559f2882add4b275d1)에 적용하는 [프론트엔드 패치](frontend.patch)와 [예측 서버](../src/serve_randomforest.py)입니다. 기존 지도, 분석 결과, 후보 비교 화면을 사용합니다. 분석 버튼은 같은 주소의 `GET /api/analysis`를 호출합니다. 창업 전략 리포트는 화면 확인용 샘플입니다.

## 실행

팀 웹앱 폴더 `AI-project-univ`가 이미 있는 상위 디렉터리에서 실행합니다. 이미 `AI_project`가 있다면 첫 줄 대신 해당 저장소의 `training-data-pipeline` 브랜치를 업데이트합니다. 브랜치에는 실행과 재학습에 필요한 최종 데이터가 포함되어 있습니다.

```bash
git clone --branch training-data-pipeline --single-branch --depth 1 https://github.com/kimwoojae1212/AI_project.git
git -C AI-project-univ apply --check ../AI_project/integration/frontend.patch
git -C AI-project-univ apply ../AI_project/integration/frontend.patch
python -m pip install numpy==2.4.4 pandas==3.0.3 scikit-learn==1.9.0 shapely==2.2.0
python AI_project/src/serve_randomforest.py --frontend AI-project-univ
```

Python 3.13을 사용합니다. 기존 웹앱의 `config.js`에 Kakao Maps JavaScript 키를 설정하고 Kakao Developers에 `http://127.0.0.1:8000`을 허용 도메인으로 추가합니다. [웹앱의 기존 설정 안내](https://github.com/woochul0516/AI-project-univ/blob/97c37f4cccd5f7d78e3225559f2882add4b275d1/README.md#실행-방법)를 참고하세요.

브라우저에서 `http://127.0.0.1:8000`을 열고 안양시의 위치와 지원 세부 업종을 고른 뒤 분석합니다. 이 서버는 로컬 통합 실행용입니다. 배포 환경에서도 웹앱과 `/api/analysis`를 같은 주소로 제공해야 합니다.

지도 키 없이 API만 확인하려면 서버 실행 후 브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8000/api/analysis?latitude=37.4075403713&longitude=126.9484580059&industry=korean-home-meal&radius=500
```

## 화면에서 보여주는 값

- 서버는 저장된 모델로 2024년 상권 대표점의 관측 입력을 예측합니다. 선택 지점에서 1km 이내의 가장 가까운 동일 업종 대표점을 사용합니다. 화면에는 해당 대표점까지의 거리를 표시합니다.
- `annual_sales`는 점포당 연매출 참고 추정치입니다. `sales`는 이를 12로 나눈 월 환산액이며 단위는 원입니다. 동종 점포 수는 대표점 반경 500m, 유동인구는 관측 행정동의 시간대 평균입니다.
- 한식·일식·중식 중 학습 데이터와 매핑되는 세부 업종과 500m 반경을 지원합니다. 분식·치킨·피자·햄버거 또는 근처에 해당 업종 관측 상권이 없는 위치는 안내 메시지를 반환합니다. [한국표준산업분류 코드 참고](https://www.krei.re.kr/consumptioninfo/page/148)
