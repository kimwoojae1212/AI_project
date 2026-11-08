import pandas as pd
from pathlib import Path
import re

data_folder = Path("data/raw")
output = Path("results/01_데이터구조확인")
output.mkdir(parents=True, exist_ok=True)

files = list(data_folder.rglob("*.csv"))
results = []

# 경기도 시군명
regions = [
    "수원시", "성남시", "안양시", "안산시",
    "용인시", "화성시", "부천시", "광명시",
    "군포시", "의왕시", "과천시", "시흥시",
    "고양시", "평택시", "김포시", "남양주시",
    "파주시", "광주시", "하남시", "오산시",
    "이천시", "안성시", "구리시", "양주시",
    "포천시", "동두천시", "여주시",
    "의정부시", "양평군", "가평군", "연천군"
]

# 1. 기존 파일명에서 지역코드와 지역명 관계 추출
code_map = {}
conflicts = set()

for file in files:
    name = file.stem

    match = re.search(
        r"_(\d{4,5})_(" + "|".join(regions) + r")$",
        name
    )

    if match:
        code, region = match.groups()

        if code in code_map and code_map[code] != region:
            conflicts.add(code)
        else:
            code_map[code] = region

# 충돌이 발생한 코드는 자동 변환하지 않음
for code in conflicts:
    code_map.pop(code, None)

print("확인된 지역코드:", code_map)
print("충돌 코드:", conflicts)

# 2. 전체 파일 분석
for file in files:
    name = file.stem

    date_match = re.search(
        r"(20\d{2})(0[1-9]|1[0-2])",
        name
    )
    year_match = re.search(r"20\d{2}", name)

    year = None
    month = None

    if date_match:
        year = int(date_match.group(1))
        month = int(date_match.group(2))
    elif year_match:
        year = int(year_match.group())

    region = "확인필요"
    method = "미확인"

    # 지역명 직접 확인
    for r in regions:
        if r in name:
            region = r
            method = "파일명"
            break

    # 지역명이 없으면 지역코드 확인
    if region == "확인필요":
        code_match = re.search(r"_(\d{4,5})$", name)

        if code_match:
            code = code_match.group(1)

            if code in code_map:
                region = code_map[code]
                method = "지역코드"

    results.append({
        "파일명": file.name,
        "데이터유형": file.parent.name,
        "지역": region,
        "연도": year,
        "월": month,
        "지역확인방법": method,
        "경로": str(file)
    })

df = pd.DataFrame(results)

# 3. 전체 파일 목록 저장
df.to_csv(
    output / "지역연도별_파일목록_수정.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. 유동인구 데이터 월별 누락 확인
population = df[
    df["파일명"].str.contains("유동인구")
    & (df["지역"] != "확인필요")
    & df["연도"].notna()
    & df["월"].notna()
].copy()

population["연월"] = (
    population["연도"].astype(int).astype(str)
    + "-"
    + population["월"].astype(int).astype(str).str.zfill(2)
)

# 2023년 1월 ~ 2025년 12월
expected_months = pd.period_range(
    "2023-01", "2025-12", freq="M"
).astype(str)

missing_results = []

for region in sorted(population["지역"].unique()):
    actual = set(
        population.loc[
            population["지역"] == region, "연월"
        ]
    )

    missing = sorted(set(expected_months) - actual)

    missing_results.append({
        "지역": region,
        "확인된월수": len(actual & set(expected_months)),
        "누락월수": len(missing),
        "누락월": ", ".join(missing) if missing else "없음"
    })

missing_df = pd.DataFrame(missing_results)

missing_df.to_csv(
    output / "유동인구_월별누락현황.csv",
    index=False,
    encoding="utf-8-sig"
)

# 5. 결과 출력
print("\n===== 전체 파일 =====")
print(len(df))

print("\n===== 지역 확인 필요 =====")
print((df["지역"] == "확인필요").sum())

print("\n===== 지역별 파일 수 =====")
print(df["지역"].value_counts())

print("\n===== 유동인구 월별 누락 현황 =====")
print(missing_df.to_string(index=False))

print("\n분석 완료!")