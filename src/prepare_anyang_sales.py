from pathlib import Path

import pandas as pd

project = Path(__file__).resolve().parents[1]
source = (
    project / "data" / "raw" / "매출"
    / "경기도 발달 골목 상권 추정 매출 현황(2025년).csv"
)
output = (
    project / "data" / "processed"
    / "안양시_확인상권2곳_음식점_추정매출_2025.csv"
)

columns = [
    "기준연도", "기준분기", "상권ID", "상권명", "행정동코드",
    "산업분류코드", "산업분류코드명", "매출금액", "매출건수", "점포수",
]

df = pd.read_csv(source, encoding="cp949", dtype="string")
df.columns = df.columns.str.strip()

missing_columns = [column for column in columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

for column in columns:
    df[column] = df[column].str.strip().replace("", pd.NA)

# 행정동코드가 비어 있으므로 확인한 상권의 ID와 이름을 함께 비교한다.
confirmed_markets = {
    "724": "안양8동주민센터",
    "25": "안양관양동우체국",
}

restaurants = df[df["산업분류코드"].str.startswith("561", na=False)]
expected_names = restaurants["상권ID"].map(confirmed_markets)
matched = restaurants["상권명"].eq(expected_names).fillna(False)
anyang = restaurants.loc[matched, columns].copy()

missing_ids = set(confirmed_markets) - set(anyang["상권ID"])
if missing_ids:
    raise ValueError(f"매칭되지 않은 상권 ID: {sorted(missing_ids)}")

key = ["기준연도", "기준분기", "상권ID", "산업분류코드"]

before = len(anyang)
anyang = anyang.drop_duplicates(subset=columns).copy()
print(f"모든 컬럼이 동일해서 제외한 행: {before - len(anyang):,}")

conflicts = anyang.duplicated(subset=key, keep=False)
if conflicts.any():
    print(anyang.loc[conflicts, columns].sort_values(key).to_string(index=False))
    raise ValueError("같은 분기·상권·업종에 서로 다른 값이 남아 있습니다.")

for column in ["매출금액", "매출건수", "점포수"]:
    anyang[column] = pd.to_numeric(anyang[column], errors="raise")

print(f"추출 행: {len(anyang):,}")
print(f"매출금액 결측: {anyang['매출금액'].isna().sum():,}")
print(f"점포수 결측: {anyang['점포수'].isna().sum():,}")

japanese = anyang.loc[
    anyang["산업분류코드명"].eq("일식 음식점업"),
    ["기준분기", "상권명", "매출금액", "점포수"],
]
print("\n일식 음식점업 행")
print(japanese.to_string(index=False))

output.parent.mkdir(parents=True, exist_ok=True)
anyang.sort_values(key).to_csv(output, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {output}")