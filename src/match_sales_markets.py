from pathlib import Path

import pandas as pd

project = Path(__file__).resolve().parents[1]
raw = project / "data" / "raw"
output = project / "results" / "04_상권연결" / "2025_매출_상권_연결후보.csv"

sales = pd.read_csv(
    raw / "매출" / "경기도 발달 골목 상권 추정 매출 현황(2025년).csv",
    encoding="cp949",
    dtype="string",
    usecols=["상권ID", "상권명"],
)
sales = sales.apply(lambda column: column.str.strip().replace("", pd.NA))
sales = sales.dropna().drop_duplicates()

catalogs = []
for filename in ["골목상권현황.csv", "발달상권현황.csv"]:
    catalog = pd.read_csv(
        raw / "상권현황" / filename,
        encoding="cp949",
        dtype="string",
        usecols=["상권명", "위도", "경도"],
    )
    catalog["상권명"] = catalog["상권명"].str.strip().replace("", pd.NA)
    catalog["현황파일"] = filename
    catalogs.append(catalog.dropna(subset=["상권명"]))

catalog = pd.concat(catalogs, ignore_index=True).drop_duplicates()
catalog["현황_동일이름수"] = catalog.groupby("상권명")["상권명"].transform("size")
sales["매출_동일이름수"] = sales.groupby("상권명")["상권명"].transform("size")
sales["매출_ID별이름수"] = sales.groupby("상권ID")["상권ID"].transform("size")

matches = sales.merge(catalog, on="상권명", how="left")
unique = (
    matches["매출_동일이름수"].eq(1)
    & matches["매출_ID별이름수"].eq(1)
    & matches["현황_동일이름수"].eq(1)
)
matches["연결상태"] = "이름 또는 ID 확인 필요"
matches.loc[matches["현황파일"].isna(), "연결상태"] = "현황에서 이름 미발견"
matches.loc[unique, "연결상태"] = "이름으로 1대1 연결 후보"

output.parent.mkdir(parents=True, exist_ok=True)
matches.sort_values(["상권명", "상권ID"]).to_csv(
    output, index=False, encoding="utf-8-sig"
)

print(matches.groupby("연결상태")["상권ID"].nunique().to_string())
print("\n기존에 확인한 두 상권")
print(
    matches.loc[
        matches["상권ID"].isin(["724", "25"]),
        ["상권ID", "상권명", "위도", "경도", "연결상태"],
    ].to_string(index=False)
)
print(f"\n저장 완료: {output}")