"""Build chronological training datasets from the repository's raw CSVs."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from build_sales_training_pilot import KEY, MARKETS

CODE = "산업분류코드"
TOTAL = "분기_상권업종_추정매출"
PER_STORE = "점포당_분기매출_참고값"
COUNT = "분기말_관측점포수"
FEATURES = ["상권ID", CODE, "기준분기", "경도", "위도"]


def read_csv(path, encoding, **kwargs):
    with path.open("rb") as source:
        if source.read(40).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Git LFS 원본을 내려받으세요: {path}")
    frame = pd.read_csv(path, encoding=encoding, dtype="string", **kwargs)
    frame.columns = frame.columns.str.strip()
    return frame


def load_sales(raw):
    columns = KEY + ["상권명", "매출금액"]
    frames = []
    for path in sorted((raw / "매출").glob("경기도 발달 골목 상권 추정 매출 현황(*년).csv")):
        frame = read_csv(path, "cp949", usecols=columns)
        frames.append(frame.apply(lambda col: col.str.strip().replace("", pd.NA)))
    if not frames:
        raise ValueError("매출 원본 CSV가 없습니다.")
    sales = pd.concat(frames, ignore_index=True)
    sales = sales.loc[sales[CODE].str.fullmatch(r"561\d{2}", na=False)].copy()
    sales = sales.dropna(subset=KEY + ["상권명"])
    for column in ["기준연도", "기준분기", "매출금액"]:
        sales[column] = pd.to_numeric(sales[column], errors="coerce")
    sales = sales.loc[
        sales["기준연도"].between(2000, 2099)
        & sales["기준분기"].isin([1, 2, 3, 4])
    ].drop_duplicates()
    sales[KEY[:2]] = sales[KEY[:2]].astype(int)
    return sales


def load_stores(path):
    columns = ["상가업소번호", "시군구명", "표준산업분류코드", "경도", "위도"]
    stores = read_csv(path, "utf-8-sig", usecols=columns)
    for column in ["경도", "위도"]:
        stores[column] = pd.to_numeric(stores[column], errors="coerce")
    stores = stores.dropna(subset=["시군구명", "상가업소번호"])
    stores = stores.loc[
        stores["경도"].between(-180, 180) & stores["위도"].between(-90, 90)
    ].drop_duplicates()
    stores["안양"] = stores["시군구명"].str.strip().str.fullmatch(r"안양시(?:\s*(?:만안구|동안구))?")
    stores[CODE] = stores["표준산업분류코드"].str.strip().str.extract(r"^I?(561\d{2})$")[0]
    return stores.reset_index(drop=True)


def store_tree(stores):
    return shapely.STRtree(shapely.points(
        stores["경도"].to_numpy(dtype=float), stores["위도"].to_numpy(dtype=float)
    ))


def load_markets(raw, sales, reference_stores):
    columns = ["상권명", "다중지역정보", "경도", "위도"]
    catalog = pd.concat([
        read_csv(raw / "상권현황" / name, "cp949", usecols=columns)
        for name in ("골목상권현황.csv", "발달상권현황.csv")
    ], ignore_index=True)
    catalog["상권명"] = catalog["상권명"].str.strip()
    for column in ["경도", "위도"]:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    catalog = catalog.dropna(subset=columns).drop_duplicates()
    catalog = catalog.loc[~catalog["상권명"].duplicated(keep=False)]
    pairs = sales[["상권ID", "상권명"]].drop_duplicates()
    unique = pairs.loc[
        ~pairs["상권ID"].duplicated(keep=False) & ~pairs["상권명"].duplicated(keep=False)
    ]
    markets = unique.merge(catalog, on="상권명", validate="one_to_one")
    markets["geometry"] = shapely.from_geojson(
        markets.pop("다중지역정보").to_numpy(), on_invalid="ignore"
    )
    markets = markets.loc[
        shapely.is_valid(markets["geometry"].to_numpy())
        & markets["geometry"].map(lambda geometry: geometry.geom_type in {"Polygon", "MultiPolygon"}
                                  if geometry is not None else False)
    ].copy()

    tree = store_tree(reference_stores)
    places = reference_stores["안양"].to_numpy(dtype=bool)
    evidence = []
    for geometry in markets["geometry"]:
        inside = places[tree.query(geometry, predicate="covers")]
        evidence.append((int(inside.sum()), int((~inside).sum())))
    markets[["안양_기준점포수", "외부_기준점포수"]] = pd.DataFrame(
        evidence, index=markets.index, columns=[0, 1]
    )
    known = markets["상권명"].eq(markets["상권ID"].map(MARKETS)).fillna(False)
    inferred = markets["안양_기준점포수"].gt(0) & markets["외부_기준점포수"].eq(0)
    markets["선정근거"] = np.select(
        [known.to_numpy(dtype=bool), inferred.to_numpy(dtype=bool)],
        ["기존 확인", "기준 점포 위치로 확인"], default="제외"
    )
    if not set(MARKETS).issubset(set(markets.loc[known, "상권ID"])):
        raise ValueError("기존 확인 상권을 매출 ID·이름과 상권 경계에 연결할 수 없습니다.")
    return markets


def count_stores(stores, markets):
    restaurants = stores.loc[stores["안양"] & stores[CODE].notna()].copy()
    restaurants = restaurants.drop_duplicates(["상가업소번호", CODE, "경도", "위도"])
    if restaurants["상가업소번호"].duplicated().any():
        raise ValueError("같은 점포 ID에 서로 다른 업종·좌표가 등록되어 있습니다.")
    tree = store_tree(restaurants)
    rows = []
    for market in markets.to_dict("records"):
        hits = tree.query(market["geometry"], predicate="covers")
        by_code = restaurants.iloc[hits][CODE].value_counts()
        rows.extend((market["상권ID"], code, count) for code, count in by_code.items())
    return pd.DataFrame(rows, columns=["상권ID", CODE, COUNT])


def split(frame):
    periods = sorted(set(zip(frame["기준연도"], frame["기준분기"])))
    if len(periods) < 2:
        raise ValueError("시간순 검증에 사용할 분기가 2개 이상 필요합니다.")
    period_keys = list(zip(frame["기준연도"], frame["기준분기"]))
    if len(periods) == 2:
        return ["test" if key == periods[-1] else "train" for key in period_keys]
    return [
        "test" if key == periods[-1] else "validation" if key == periods[-2] else "train"
        for key in period_keys
    ]


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.project / "data/raw"
    output = args.output or args.project / "data/processed/training"

    sales = load_sales(raw)
    snapshots = {}
    for path in sorted((raw / "상권").glob("소상공인시장진흥공단_상가(상권)정보_경기_*.csv")):
        match = re.search(r"_(\d{4})(03|06|09|12)$", path.stem)
        if match:
            year, month = map(int, match.groups())
            snapshots[(year, month // 3)] = path
    if not snapshots:
        raise ValueError("분기 말 점포 스냅샷이 없습니다.")
    reference_period = min(snapshots)
    reference = load_stores(snapshots[reference_period])
    if not reference["안양"].any():
        raise ValueError("기준 점포 파일에 안양시 상가가 없습니다.")
    audit_markets = load_markets(raw, sales, reference)
    markets = audit_markets.loc[audit_markets["선정근거"].ne("제외")].copy()
    context = markets[["상권ID", "상권명", "경도", "위도"]]
    matched = sales.merge(context, on=["상권ID", "상권명"], validate="many_to_one")
    if matched.duplicated(KEY).any():
        raise ValueError("같은 분기·상권·업종에 서로 다른 매출이 있습니다.")
    valid = matched["매출금액"].ge(0) & np.isfinite(matched["매출금액"])
    total = matched.loc[valid].rename(columns={"매출금액": TOTAL})
    total = total[KEY + ["경도", "위도", TOTAL]].sort_values(KEY).copy()
    total["split"] = split(total)

    per_store_periods = []
    audit_periods = []
    periods = set(snapshots) | set(zip(sales["기준연도"], sales["기준분기"]))
    for year, quarter in sorted(periods):
        rows = total.loc[total["기준연도"].eq(year) & total["기준분기"].eq(quarter)]
        snapshot = snapshots.get((year, quarter))
        count_valid = 0
        if snapshot is not None and not rows.empty:
            shops = reference if (year, quarter) == reference_period else load_stores(snapshot)
            counts = count_stores(shops, markets)
            joined = rows.drop(columns="split").merge(
                counts, on=["상권ID", CODE], validate="many_to_one"
            )
            joined[PER_STORE] = joined.pop(TOTAL) / joined[COUNT]
            count_valid = len(joined)
            per_store_periods.append(joined)
        audit_periods.append((year, quarter, len(rows), snapshot.name if snapshot else "", count_valid))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(total, output / "quarterly_market_sales.csv")
    if per_store_periods:
        per_store = pd.concat(per_store_periods, ignore_index=True).sort_values(KEY)
        observed_periods = per_store[KEY[:2]].drop_duplicates()
        if len(observed_periods) >= 2:
            per_store["split"] = split(per_store)
            write_csv(per_store, output / "quarterly_per_store_sales.csv")
        else:
            raise ValueError("점포당 학습 데이터에 같은 분기의 매출·점포가 2개 기간 이상 필요합니다.")
    else:
        raise ValueError("같은 분기에 매출과 점포 파일이 연결되지 않았습니다.")
    write_csv(audit_markets.drop(columns="geometry"), output / "market_audit.csv")
    write_csv(pd.DataFrame(audit_periods, columns=[
        "기준연도", "기준분기", "유효_상권업종_매출행", "점포_스냅샷", "유효_점포당_매출행"
    ]), output / "period_audit.csv")
    summary = {
        "reference_store_period": reference_period,
        "markets": len(markets),
        "invalid_matched_sales": int((~valid).sum()),
        "datasets": {
            "quarterly_market_sales.csv": {"features": FEATURES, "target": TOTAL,
                                           "split_counts": total["split"].value_counts().to_dict()},
            "quarterly_per_store_sales.csv": {"features": FEATURES, "target": PER_STORE,
                                              "split_counts": per_store["split"].value_counts().to_dict()},
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
