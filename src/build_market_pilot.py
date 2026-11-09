import argparse
from pathlib import Path

import pandas as pd
import shapely

# 앱 연동용 전처리
def main():
    parser = argparse.ArgumentParser(description="안양 두 상권의 일식 매출과 점포 수를 연결합니다.")
    parser.add_argument("project", type=Path, help="AI_project 폴더")
    parser.add_argument("output", type=Path, help="저장할 CSV 경로")
    args = parser.parse_args()
    data = args.project / "data"

    sales = pd.read_csv(
        data / "processed" / "안양시_확인상권2곳_음식점_추정매출_2025.csv",
        dtype="string",
    )
    sales = sales.loc[sales["산업분류코드"].eq("56122"), [
        "상권ID", "상권명", "산업분류코드명", "기준연도", "기준분기", "매출금액",
    ]].copy()
    if len(sales) != 2 or set(sales["상권ID"]) != {"724", "25"}:
        raise ValueError("두 상권의 일식 매출이 각각 한 행씩 필요합니다.")

    stores = pd.read_csv(
        data / "raw" / "상권" / "소상공인시장진흥공단_상가(상권)정보_경기_202512.csv",
        dtype="string",
        usecols=["상가업소번호", "시군구명", "표준산업분류코드", "경도", "위도"],
    )
    stores = stores.loc[
        stores["시군구명"].str.startswith("안양시", na=False)
        & stores["표준산업분류코드"].eq("I56122")
    ].dropna(subset=["상가업소번호", "경도", "위도"])
    stores = stores.drop_duplicates(subset=["상가업소번호", "경도", "위도"])
    if stores["상가업소번호"].duplicated().any():
        raise ValueError("같은 점포 ID에 서로 다른 좌표가 있습니다.")
    points = shapely.points(stores["경도"].astype(float), stores["위도"].astype(float))

    areas = pd.read_csv(data / "raw" / "상권현황" / "발달상권현황.csv", encoding="cp949")
    areas = areas.loc[areas["상권명"].isin(sales["상권명"]), ["상권명", "다중지역정보"]]
    result = sales.merge(areas, on="상권명", how="left", validate="one_to_one")
    if result["다중지역정보"].isna().any():
        raise ValueError("상권 경계를 찾을 수 없습니다.")
    result["경계내_일식점포수"] = result.pop("다중지역정보").map(
        lambda boundary: int(shapely.covers(shapely.from_geojson(boundary), points).sum())
    )
    result["점포수_기준연월"] = "202512"
    result["점포당_추정매출"] = pd.NA
    result["점포당매출_미산출사유"] = "매출과 점포수의 기준 시점이 다름"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()