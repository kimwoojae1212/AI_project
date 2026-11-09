from pathlib import Path

import geopandas as gpd
import pandas as pd

project = Path(__file__).resolve().parents[1]
raw = project / "data" / "raw"
candidate_file = (
    project / "results" / "04_상권연결" / "2025_매출_상권_연결후보.csv"
)
boundary_file = raw / "경계" / "안양시_경계.geojson"
sales_file = (
    raw / "매출" / "경기도 발달 골목 상권 추정 매출 현황(2025년).csv"
)
output = project / "data" / "processed" / "안양시_연결상권_음식점_추정매출_2025.csv"

if not boundary_file.is_file():
    raise FileNotFoundError(f"안양시 경계 파일이 필요합니다: {boundary_file}")

# 1. 이름으로 1대1 연결된 후보의 대표 좌표를 경계와 비교
boundary = gpd.read_file(boundary_file)
if boundary.empty or boundary.crs is None:
    raise ValueError("경계 파일에 도형과 좌표계 정보가 필요합니다.")

candidates = pd.read_csv(candidate_file, encoding="utf-8-sig", dtype="string")
unique = candidates.loc[
    candidates["연결상태"].eq("이름으로 1대1 연결 후보"),
    ["상권ID", "상권명", "위도", "경도"],
].copy()

for column in ["위도", "경도"]:
    unique[column] = pd.to_numeric(unique[column], errors="coerce")

valid = unique["위도"].between(-90, 90) & unique["경도"].between(-180, 180)
valid = valid.fillna(False)
points = unique.loc[valid].copy()
points = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(points["경도"], points["위도"]),
    crs="EPSG:4326",
)

city = boundary.to_crs("EPSG:4326").geometry.union_all()
markets = points.loc[points.geometry.intersects(city), unique.columns].copy()

# 앞서 확인한 두 곳으로 경계 파일과 좌표를 점검
missing_known = {"724", "25"} - set(markets["상권ID"])
if missing_known:
    raise ValueError(f"기존 안양 상권이 경계 밖으로 판정됐습니다: {sorted(missing_known)}")

# 2. 안양 상권의 음식점 매출 연결
sales = pd.read_csv(sales_file, encoding="cp949", dtype="string")
sales.columns = sales.columns.str.strip()
for column in sales.columns:
    sales[column] = sales[column].str.strip().replace("", pd.NA)

restaurants = sales.loc[sales["산업분류코드"].str.startswith("561", na=False)]
matched = restaurants.merge(
    markets, on=["상권ID", "상권명"], how="inner", validate="many_to_one"
)

# 3. 완전히 같은 행은 한 건으로 줄이고, 같은 키에 다른 값이 있으면 중단
cleaned = matched.drop_duplicates().copy()
key = ["기준연도", "기준분기", "상권ID", "산업분류코드"]
conflicts = cleaned.duplicated(subset=key, keep=False)

if conflicts.any():
    print(cleaned.loc[conflicts].sort_values(key).head(10).to_string(index=False))
    raise ValueError("같은 분기·상권·업종 조합에 서로 다른 값이 있습니다.")

if cleaned.empty:
    raise ValueError("안양 상권과 연결된 음식점 매출 행이 없습니다.")

for column in ["매출금액", "매출건수", "점포수"]:
    cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

output.parent.mkdir(parents=True, exist_ok=True)
cleaned.sort_values(key).to_csv(output, index=False, encoding="utf-8-sig")

print(f"좌표가 유효하지 않은 1대1 후보: {(~valid).sum():,}개")
print(f"안양 경계 안에 있는 연결 상권: {len(markets):,}개")
print(f"완전히 동일해서 제외한 행: {len(matched) - len(cleaned):,}개")
print(f"최종 매출 행: {len(cleaned):,}개")
print(f"점포수 결측: {cleaned['점포수'].isna().sum():,}개")
print(f"저장 완료: {output}")