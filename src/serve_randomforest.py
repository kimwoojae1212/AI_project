"""Serve the competition frontend and the saved Random Forest on one origin."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data/processed/training"
CODE = "산업분류코드"
COORDS = ["위도", "경도"]
MONTHS_PER_YEAR = 12
MAX_DISTANCE_M = 1000
EARTH_RADIUS_M = 6371008.8
# 세부 메뉴를 학습 데이터의 한국표준산업분류 단위로 연결한다.
INDUSTRY_CODES = {
    "korean-home-meal": "56111",
    "korean-soup": "56111",
    "korean-grill": "56113",
    "korean-jokbal-bossam": "56113",
    "japanese-sushi": "56122",
    "japanese-tonkatsu": "56122",
    "japanese-ramen-udon": "56122",
    "chinese-jajang-jjamppong": "56121",
    "chinese-malatang": "56121",
}


class Predictor:
    def __init__(self, frontend):
        boundary = frontend / "data/geojson/anyang-si.geojson"
        geojson = json.loads(boundary.read_text(encoding="utf-8-sig"))
        self.area = unary_union([shape(item["geometry"]) for item in geojson["features"]])
        manifest = json.loads((TRAINING / "annual_training_manifest.json").read_text(encoding="utf-8-sig"))
        self.features = manifest["feature_columns"]
        data = pd.read_csv(TRAINING / "annual_per_store_training.csv", dtype={CODE: object})
        self.year = int(data["기준연도"].max())
        self.data = data.loc[data["기준연도"].eq(self.year)].reset_index(drop=True)
        with (ROOT / "results/randomforest_model.pkl").open("rb") as source:
            self.model = pickle.load(source)
        trained_codes = self.model.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ].named_steps["encoder"].categories_[0]
        if not set(INDUSTRY_CODES.values()).issubset(set(trained_codes) & set(self.data[CODE])):
            raise ValueError("업종 매핑과 학습 모델 또는 관측 자료가 일치하지 않습니다.")
        # 입력은 고정된 관측 행이므로 모든 대표점의 모델 추론을 시작할 때 한 번만 수행한다.
        self.annual_sales = self.model.predict(self.data[self.features])

    def predict(self, latitude, longitude, industry, radius):
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("유효한 지도 위치를 선택해 주세요.")
        if radius != 500:
            raise ValueError("랜덤 포레스트 예측은 500m 관측 자료를 사용합니다. 500m를 선택해 주세요.")
        code = INDUSTRY_CODES.get(industry)
        if code is None:
            raise ValueError("선택한 업종의 학습 자료가 없습니다. 한식·일식·중식의 지원 업종을 선택해 주세요.")
        if not self.area.covers(Point(longitude, latitude)):
            raise ValueError("안양시 안의 위치를 선택해 주세요.")

        candidates = self.data.loc[self.data[CODE].eq(code)]
        points = np.radians(candidates[COORDS].to_numpy(dtype=float))
        lat, lon = np.radians([latitude, longitude])
        delta_lat = (points[:, 0] - lat) / 2
        delta_lon = (points[:, 1] - lon) / 2
        haversine = np.sin(delta_lat)**2 + np.cos(lat) * np.cos(points[:, 0]) * np.sin(delta_lon)**2
        distances = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        nearest = int(distances.argmin())
        if distances[nearest] > MAX_DISTANCE_M:
            raise ValueError("1km 안에 같은 업종의 관측 상권이 없습니다. 다른 위치를 선택해 주세요.")
        row = candidates.iloc[nearest]
        annual = float(self.annual_sales[row.name])
        return {
            "status": "success",
            "sales": annual / MONTHS_PER_YEAR,
            "annual_sales": annual,
            "competitor_count": int(row["500m_동종점포수"]),
            "foot_traffic": float(row["행정동_시간대평균_유동인구"]),
            "accessibility_score": None,
            "reference_year": self.year,
            "reference_distance_m": round(float(distances[nearest])),
            "radius": radius,
            "mock": False,
        }


class Handler(SimpleHTTPRequestHandler):
    web_paths = {
        "/", "/index.html", "/styles.css", "/config.js", "/app.js",
        "/analysis-model.mjs", "/geo-boundary.mjs",
        "/data/geojson/anyang-si.geojson", "/data/geojson/anyang-commercial-areas.geojson",
    }

    def send_head(self):
        path = urlsplit(self.path).path
        target = Path(self.translate_path(path)).resolve()
        if path not in self.web_paths or not target.is_relative_to(Path(self.directory)):
            self.send_error(404)
            return None
        return super().send_head()

    def __init__(self, *args, predictor, **kwargs):
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def do_GET(self):
        request = urlsplit(self.path)
        if request.path != "/api/analysis":
            return super().do_GET()

        params = parse_qs(request.query)
        status = 200
        try:
            latitude = float(params["latitude"][0])
            longitude = float(params["longitude"][0])
            industry = params["industry"][0]
            radius = int(params["radius"][0])
        except (KeyError, ValueError):
            status, result = 400, {"error": "좌표, 업종 및 반경을 확인해 주세요."}
        else:
            try:
                result = self.predictor.predict(latitude, longitude, industry, radius)
            except ValueError as error:
                status, result = 422, {"error": str(error)}

        body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", required=True, type=Path, help="패치를 적용한 웹앱 경로")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    frontend = args.frontend.resolve()
    if not (frontend / "index.html").is_file():
        parser.error("--frontend에 웹앱 루트 경로를 지정하세요.")
    handler = partial(Handler, directory=str(frontend), predictor=Predictor(frontend))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"http://{args.host}:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
