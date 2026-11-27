"""Build the annual per-store revenue reference and nearby restaurant features."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from build_training_data import CODE, TOTAL, count_stores, load_markets, load_sales, load_stores

RADIUS_KM = 6371.0088
NEIGHBORHOOD_M = 500
COORDS = ["위도", "경도"]
NEARBY = ["500m_음식점수", "500m_동종점포수", "동종점포_최근거리_m_상한500"]
FEATURES = [CODE, *COORDS, *NEARBY]
TARGET = "점포당_연매출_참고값"


class NearbyRestaurants:
    def __init__(self, stores):
        shops = stores.loc[stores[CODE].notna()].drop_duplicates(["상가업소번호", CODE, *COORDS])
        if shops["상가업소번호"].duplicated().any():
            raise ValueError("점포 ID에 서로 다른 업종 또는 좌표가 있습니다.")
        self.codes = shops[CODE].to_numpy(dtype=str)
        if not len(shops):
            raise ValueError("음식점 좌표가 없습니다.")
        self.tree = BallTree(np.radians(shops[COORDS].to_numpy(dtype=float)), metric="haversine")

    def features(self, frame):
        points = np.radians(frame[COORDS].to_numpy(dtype=float))
        indices, distances = self.tree.query_radius(
            points, r=NEIGHBORHOOD_M / (RADIUS_KM * 1000), return_distance=True
        )
        rows = []
        for code, hits, radians in zip(frame[CODE], indices, distances):
            same = self.codes[hits] == code
            nearest = radians[same].min() * RADIUS_KM * 1000 if same.any() else NEIGHBORHOOD_M
            rows.append((len(hits), int(same.sum()), float(nearest)))
        return pd.DataFrame(rows, columns=NEARBY, index=frame.index)


def find_snapshots(raw):
    snapshots = {}
    for path in sorted((raw / "상권").glob("소상공인시장진흥공단_상가(상권)정보_경기_*.csv")):
        match = re.search(r"_(\d{4})(03|06|09|12)$", path.stem)
        if match:
            year, month = map(int, match.groups())
            snapshots[(year, month // 3)] = path
    if not snapshots:
        raise ValueError("분기 말 점포 스냅샷이 없습니다.")
    return snapshots


def build_dataset(project, output, proxy):
    raw = project / "data/raw"
    source = project / "data/processed/training/quarterly_market_sales.csv"
    quarterly = pd.read_csv(source, dtype={"상권ID": str, CODE: str})
    snapshots = find_snapshots(raw)
    reference = load_stores(snapshots[min(snapshots)])
    markets = load_markets(raw, load_sales(raw), reference)
    markets = markets.loc[markets["선정근거"].ne("제외")]

    frames, audit = [], []
    key = ["상권ID", CODE]
    for year, quarters in quarterly.groupby("기준연도"):
        sales_quarters = sorted(quarters["기준분기"].unique().tolist())
        available = [q for q in range(1, 5) if (year, q) in snapshots]
        rows = 0
        if sales_quarters == [1, 2, 3, 4] and (available if proxy else len(available) == 4):
            selected = available[:1] if proxy else available
            counts = []
            for quarter in selected:
                path = snapshots[(year, quarter)]
                stores = reference if (year, quarter) == min(snapshots) else load_stores(path)
                counts.append(count_stores(stores, markets).assign(기준분기=quarter))
                if quarter == selected[0]:
                    nearby = NearbyRestaurants(stores)
            counts = pd.concat(counts, ignore_index=True)
            if proxy:
                joined = quarters.merge(counts.drop(columns="기준분기"), on=key, validate="many_to_one")
            else:
                joined = quarters.merge(counts, on=[*key, "기준분기"], validate="one_to_one")
            joined[TARGET] = joined[TOTAL] / joined["분기말_관측점포수"]
            annual = joined.groupby(key, as_index=False).agg(
                **{TARGET: (TARGET, "sum"), "분기수": ("기준분기", "nunique"),
                   "위도": ("위도", "first"), "경도": ("경도", "first")}
            )
            annual = annual.loc[annual["분기수"].eq(4)].drop(columns="분기수").copy()
            annual[NEARBY] = nearby.features(annual)
            annual["기준연도"] = year
            annual["특성_점포기준연월"] = f"{year}{selected[0] * 3:02d}"
            rows = len(annual)
            if rows:
                frames.append(annual)
        audit.append({"기준연도": int(year), "매출_분기": sales_quarters, "점포_분기": available, "사용_행": rows})
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if len(frames) < 2:
        raise ValueError("시간순 학습·평가에 유효한 연도 2개가 필요합니다. 분기별 원본 또는 --proxy를 확인하세요.")
    data = pd.concat(frames, ignore_index=True)
    data["split"] = np.where(data["기준연도"].eq(data["기준연도"].max()), "test", "train")
    data["목표값_산정방식"] = "연간총매출÷연중단일시점점포수" if proxy else "분기별점포당매출의4분기합"
    output.parent.mkdir(parents=True, exist_ok=True)
    data[["기준연도", "상권ID", *FEATURES, TARGET, "split", "특성_점포기준연월", "목표값_산정방식"]].to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    print(f"연매출 학습 데이터 저장: {output} ({len(data):,}행)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--proxy", action="store_true", help="연중 관측한 한 시점의 점포 수 사용")
    args = parser.parse_args()
    filename = "annual_per_store_proxy.csv" if args.proxy else "annual_per_store.csv"
    output = args.output or args.project / "data/processed/training" / filename
    build_dataset(args.project, output, args.proxy)


if __name__ == "__main__":
    main()
