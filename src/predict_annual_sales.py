"""Build and serve a map-based estimate of annual sales per restaurant."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from build_training_data import CODE, TOTAL, count_stores, load_markets, load_sales, load_stores

RADIUS_KM = 6371.0088
NEIGHBORHOOD_M = 500
MAX_SUPPORT_M = 1000
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


def fit_model(path):
    data = pd.read_csv(path, dtype={CODE: str, "상권ID": str})
    train, test = (data.loc[data["split"].eq(part)] for part in ("train", "test"))
    encode = ColumnTransformer([
        ("업종", OneHotEncoder(handle_unknown="ignore"), [CODE]),
        ("수치", "passthrough", [*COORDS, *NEARBY]),
    ])
    model = make_pipeline(encode, RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(train[FEATURES], train[TARGET])
    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_mae_won": round(mean_absolute_error(test[TARGET], model.predict(test[FEATURES]))),
        "test_median_baseline_mae_won": round(mean_absolute_error(
            test[TARGET], np.full(len(test), train[TARGET].median())
        )),
    }
    return model, train, metrics


def load_predictor(project, dataset):
    model, train, _ = fit_model(dataset)
    snapshots = find_snapshots(project / "data/raw")
    latest = max(snapshots)
    nearby = NearbyRestaurants(load_stores(snapshots[latest]))
    method = pd.read_csv(dataset, usecols=["목표값_산정방식"], nrows=1).iloc[0, 0]
    return model, train, method, latest, nearby


def predict(lat, lon, code, predictor):
    model, train, method, latest, nearby = predictor
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("위도·경도 범위를 확인하세요.")
    candidates = train.loc[train[CODE].eq(code)]
    if candidates.empty:
        raise ValueError("해당 업종의 학습 자료가 없습니다.")
    coordinates = np.radians([[lat, lon]])
    support = BallTree(np.radians(candidates[COORDS].to_numpy(dtype=float)), metric="haversine")
    distance, nearest = support.query(coordinates, k=1)
    distance_m = float(distance[0, 0] * RADIUS_KM * 1000)
    if distance_m > MAX_SUPPORT_M:
        raise ValueError(f"가장 가까운 동종 업종 학습 상권까지 {distance_m:.0f}m로, 예측 지원 범위 밖입니다.")

    query = pd.DataFrame([{CODE: code, "위도": lat, "경도": lon}])
    query[NEARBY] = nearby.features(query)
    amount = model.predict(query[FEATURES])[0]
    return {
        "점포당_예상연매출_참고값_원": round(float(amount)),
        "산정방식": method,
        "업종코드": code,
        "위도": lat,
        "경도": lon,
        "주변점포_기준연월": f"{latest[0]}{latest[1] * 3:02d}",
        "가장_가까운_동종_학습상권ID": candidates.iloc[nearest[0, 0]]["상권ID"],
        "가장_가까운_동종_학습상권_대표점_거리_m": round(distance_m),
        **{name: round(float(query.at[0, name])) for name in NEARBY},
    }


def serve(predictor, port):
    def app(environ, start_response):
        status = "200 OK"
        if environ["REQUEST_METHOD"] != "GET" or environ["PATH_INFO"] != "/predict":
            status, result = "404 Not Found", {"error": "GET /predict 경로를 사용하세요."}
        else:
            params = parse_qs(environ["QUERY_STRING"])
            try:
                lat = float(params["lat"][0])
                lon = float(params["lon"][0])
                code = params["code"][0]
                result = predict(lat, lon, code, predictor)
            except KeyError:
                status, result = "400 Bad Request", {"error": "lat, lon, code가 필요합니다."}
            except ValueError as error:
                status, result = "400 Bad Request", {"error": str(error)}
        start_response(status, [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps(result, ensure_ascii=False).encode("utf-8")]

    with make_server("127.0.0.1", port, app) as server:
        print(f"로컬 연매출 API: http://127.0.0.1:{port}/predict", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path)
    command = parser.add_subparsers(dest="command", required=True)
    build = command.add_parser("build")
    build.add_argument("--proxy", action="store_true", help="연중 한 시점의 분모를 쓰는 데모용 CSV")
    command.add_parser("evaluate")
    server = command.add_parser("serve")
    server.add_argument("--port", type=int, default=8000)
    forecast = command.add_parser("predict")
    forecast.add_argument("--lat", type=float, required=True)
    forecast.add_argument("--lon", type=float, required=True)
    forecast.add_argument("--code", required=True)
    args = parser.parse_args()

    folder = args.project / "data/processed/training"
    if args.command == "build":
        filename = "annual_per_store_proxy.csv" if args.proxy else "annual_per_store.csv"
        build_dataset(args.project, args.dataset or folder / filename, args.proxy)
        return
    dataset = args.dataset or folder / "annual_per_store_proxy.csv"
    if args.command == "evaluate":
        _, _, result = fit_model(dataset)
    else:
        predictor = load_predictor(args.project, dataset)
        if args.command == "serve":
            serve(predictor, args.port)
            return
        result = predict(args.lat, args.lon, args.code, predictor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
