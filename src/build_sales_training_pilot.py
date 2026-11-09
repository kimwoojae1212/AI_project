
import argparse
from pathlib import Path

import pandas as pd
import shapely

MARKETS = {"724": "안양8동주민센터", "25": "안양관양동우체국"}
PERIODS = [("2023", "1", "202303"), ("2024", "3", "202409")]
KEY = ["기준연도", "기준분기", "상권ID", "산업분류코드"]


def build_period(raw, boundaries, year, quarter, month):
    sales = pd.read_csv(
        raw / "매출" / f"경기도 발달 골목 상권 추정 매출 현황({year}년).csv",
        encoding="cp949", dtype="string",
    )
    sales = sales.loc[
        sales["기준연도"].eq(year)
        & sales["기준분기"].eq(quarter)
        & sales["상권명"].eq(sales["상권ID"].map(MARKETS))
        & sales["산업분류코드"].str.startswith("561", na=False)
    ].drop_duplicates().copy()
    if sales.empty or sales.duplicated(KEY).any():
        raise ValueError(f"{year}년 {quarter}분기 매출 누락 또는 키 충돌")

    path = raw / "상권" / f"소상공인시장진흥공단_상가(상권)정보_경기_{month}.csv"
    columns = ["상가업소번호", "시군구명", "표준산업분류코드", "경도", "위도"]
    stores = pd.read_csv(path, dtype="string", usecols=columns, encoding="utf-8-sig")
    stores = stores.loc[stores["시군구명"].str.startswith("안양시", na=False)].copy()
    stores["산업분류코드"] = stores["표준산업분류코드"].str.extract(r"^I?(561\d{2})$")[0]
    stores = stores.dropna(subset=["산업분류코드", "상가업소번호", "경도", "위도"])
    stores = stores.drop_duplicates(["상가업소번호", "산업분류코드", "경도", "위도"])
    if stores["상가업소번호"].duplicated().any():
        raise ValueError(f"{month} 점포 ID에 업종 또는 좌표 충돌")
    points = shapely.points(stores["경도"].astype(float), stores["위도"].astype(float))

    counts = []
    for market_id, name in MARKETS.items():
        inside = shapely.covers(boundaries[name], points)
        by_code = stores.loc[inside, "산업분류코드"].value_counts()
        for code, count in by_code.items():
            counts.append((market_id, code, count))
    counts = pd.DataFrame(counts, columns=["상권ID", "산업분류코드", "분기말_관측점포수"])
    result = sales.merge(counts, on=["상권ID", "산업분류코드"], how="left", validate="many_to_one")
    count = result["분기말_관측점포수"].fillna(0).astype(int)
    result["분기말_관측점포수"] = count
    result["점포수_기준연월"] = month
    amount = pd.to_numeric(result["매출금액"], errors="raise")
    result["점포당_분기매출_참고값"] = amount.where(amount.ge(0)) / count.where(count.gt(0))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.project / "data" / "raw"
    catalog = pd.read_csv(raw / "상권현황" / "발달상권현황.csv", encoding="cp949")
    catalog = catalog.loc[catalog["상권명"].isin(MARKETS.values())]
    if len(catalog) != len(MARKETS) or set(catalog["상권명"]) != set(MARKETS.values()):
        raise ValueError("확인한 상권마다 경계가 하나씩 있어야 합니다.")
    boundaries = dict(zip(catalog["상권명"], shapely.from_geojson(catalog["다중지역정보"])))
    result = pd.concat(
        [build_period(raw, boundaries, *period) for period in PERIODS], ignore_index=True,
    ).sort_values(KEY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    valid = result["점포당_분기매출_참고값"].notna()
    print(result.assign(참고값있음=valid).groupby(["기준연도", "기준분기"])["참고값있음"].agg(["size", "sum"]))
    print(result.loc[result["산업분류코드"].eq("56122"),
          KEY + ["분기말_관측점포수", "점포당_분기매출_참고값"]].to_string(index=False))


if __name__ == "__main__":
    main()