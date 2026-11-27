"""Prepare the shared annual restaurant revenue dataset and population features."""
import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

from build_training_data import CODE, COUNT, TOTAL
from build_annual_training_source import FEATURES, TARGET

DONG = "행정동코드"
MONTH = "특성_점포기준연월"
POPULATION_FEATURES = ["행정동_시간대평균_유동인구", "행정동_저녁시간대평균_유동인구", "행정동_유동인구_20대30대_비율"]
AGE_COLUMNS = [f"{sex}_{age}_CNT" for sex in ("M", "F") for age in range(10, 71, 5)]
YOUNG_COLUMNS = [name for name in AGE_COLUMNS if 20 <= int(name.split("_")[1]) < 40]
POP_KEYS = ["ADMI_CD", "ETL_YMD", "TIME_CD", "FORN_GB"]

# 안양시 조례 제3562호, 제3760호, 제3853호의 명칭 변경을 2026 경계에 맞춘다.
HISTORICAL_DONG_CODES = {
    ("41171", "안양8동"): "41171582",
    ("41171", "안양9동"): "41171583",
    ("41171", "석수3동"): "41171611",
    ("41171", "박달1동"): "41171640",
    ("41171", "박달2동"): "41171650",
    ("41173", "관양1동"): "41173552",
    ("41173", "관양2동"): "41173559",
}


class AnyangDistricts:
    def __init__(self, path):
        features = json.loads(path.read_text(encoding="utf-8"))["features"]
        self.codes = []
        polygons = []
        for feature in features:
            official = str(feature["properties"]["adm_cd2"])
            if not (len(official) == 10 and official.isdigit() and official.startswith(("41171", "41173"))):
                raise ValueError(f"안양시 행정동 코드가 아닙니다: {official}")
            polygon = shape(feature["geometry"])
            if polygon.is_empty or not polygon.is_valid or polygon.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"유효하지 않은 행정동 경계: {official}")
            self.codes.append(official[:8])
            polygons.append(polygon)
        if not polygons or len(set(self.codes)) != len(self.codes):
            raise ValueError("안양시 행정동 경계가 비어 있거나 코드가 중복됩니다.")
        self.tree = shapely.STRtree(polygons)

    def codes_at(self, locations):
        points = shapely.points(locations["경도"].to_numpy(), locations["위도"].to_numpy())
        codes = []
        for point in points:
            hits = self.tree.query(point, predicate="covered_by")
            codes.append(self.codes[hits[0]] if len(hits) == 1 else None)
        return pd.Series(codes, index=locations.index, dtype="string", name=DONG)


def aggregate_population(path, months):
    with path.open("rb") as source:
        if source.read(40).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("유동인구 CSV의 Git LFS 원본을 내려받으세요.")
    columns = [*POP_KEYS, "ADMI_NM", "기준연월", *AGE_COLUMNS]
    parts = []
    for chunk in pd.read_csv(
        path, usecols=columns, chunksize=100_000,
        dtype={"ADMI_CD": "string", "기준연월": "string"}
    ):
        chunk = chunk.loc[chunk["기준연월"].isin(months)]
        if chunk.empty:
            continue
        if not np.isfinite(chunk[AGE_COLUMNS].to_numpy(dtype=float)).all() or (chunk[AGE_COLUMNS] < 0).any().any():
            raise ValueError("유동인구에 결측·음수 또는 무한대가 있습니다.")
        values = chunk[["기준연월", *POP_KEYS, "ADMI_NM"]].copy()
        values["인구"] = chunk[AGE_COLUMNS].sum(axis=1)
        values["20대30대"] = chunk[YOUNG_COLUMNS].sum(axis=1)
        parts.append(values)
    if not parts:
        raise ValueError("지정한 월의 유동인구가 없습니다.")
    raw = pd.concat(parts, ignore_index=True)
    if raw[POP_KEYS].isna().any().any() or raw.duplicated(POP_KEYS).any():
        raise ValueError("유동인구 날짜·행정동·시간·내외국인 키가 비어 있거나 중복됩니다.")
    if not raw["TIME_CD"].between(0, 23).all():
        raise ValueError("유동인구 시간대 범위가 올바르지 않습니다.")
    key = ["기준연월", "ADMI_CD"]
    hourly = raw.groupby([*key, "ETL_YMD", "TIME_CD"], as_index=False)[["인구", "20대30대"]].sum()
    grouped = hourly.groupby(key)
    out = grouped.agg(
        행정동_시간대평균_유동인구=("인구", "mean"),
        관측시간수=("TIME_CD", "size"),
        관측일수=("ETL_YMD", "nunique"),
    )
    evening = hourly.loc[hourly["TIME_CD"].between(17, 21)].groupby(key)
    out[POPULATION_FEATURES[1]] = evening["인구"].mean()
    totals = grouped[["인구", "20대30대"]].sum()
    out[POPULATION_FEATURES[2]] = totals["20대30대"].div(totals["인구"].where(totals["인구"].gt(0)))
    out = out.reset_index().rename(columns={"기준연월": MONTH, "ADMI_CD": DONG})
    names = raw[["기준연월", "ADMI_CD", "ADMI_NM"]].drop_duplicates().rename(
        columns={"기준연월": MONTH, "ADMI_CD": DONG}
    )
    out = out.merge(names, on=[MONTH, DONG], validate="one_to_one")
    out["유동인구_원본행정동코드"] = out[DONG]
    out[DONG] = [
        HISTORICAL_DONG_CODES.get((code[:5], name.strip()), code)
        for code, name in zip(out[DONG], out["ADMI_NM"])
    ]
    if out.duplicated([MONTH, DONG]).any():
        raise ValueError("행정동 명칭 변경 적용 후 월·행정동 코드가 중복됩니다.")
    out["월_달력시간수"] = out[MONTH].map(lambda m: calendar.monthrange(int(m[:4]), int(m[4:]))[1] * 24)
    return out


def attach_population(locations, districts, population):
    data = locations.copy()
    data[DONG] = districts.codes_at(data)
    return data.merge(population, on=[DONG, MONTH], how="left", validate="many_to_one")


def audit_target(data, folder):
    keys = ["기준연도", "상권ID", CODE]
    dtype = {"상권ID": "string", CODE: "string"}
    quarterly = pd.read_csv(folder / "quarterly_market_sales.csv", dtype=dtype)
    annual = quarterly.groupby(keys, as_index=False).agg(
        추정매출4분기합=(TOTAL, "sum"), 포함분기수=("기준분기", "nunique")
    )
    stores = pd.read_csv(folder / "quarterly_per_store_sales.csv", dtype=dtype, usecols=[*keys, COUNT, "기준분기"])
    stores[MONTH] = stores["기준연도"].astype(str) + (stores.pop("기준분기") * 3).map("{:02d}".format)
    matched = data.merge(annual, on=keys, validate="one_to_one").merge(
        stores, on=[*keys, MONTH], validate="one_to_one"
    )
    if len(matched) != len(data) or not matched["포함분기수"].eq(4).all() or not matched[COUNT].gt(0).all():
        raise ValueError("연매출 정답의 원본 매출 또는 분모를 추적할 수 없습니다.")
    if not np.allclose(matched[TARGET], matched["추정매출4분기합"] / matched[COUNT], rtol=0, atol=1e-5):
        raise ValueError("연매출 정답 검산에 실패했습니다.")
    return matched


def prepare(folder, output, population_path, boundary_path, serving_month):
    dtype = {"상권ID": "string", CODE: "string", MONTH: "string"}
    source = pd.read_csv(folder / "annual_per_store_proxy.csv", dtype=dtype)
    train_months = set(source[MONTH])
    districts = AnyangDistricts(boundary_path)
    population = aggregate_population(population_path, train_months | {serving_month})
    if serving_month not in set(population[MONTH]):
        raise ValueError("서비스에 사용할 월의 유동인구를 찾을 수 없습니다.")
    audited = audit_target(source, folder)
    data = attach_population(audited, districts, population)
    data["유동인구_연결"] = data[POPULATION_FEATURES].notna().all(axis=1)
    key = ["기준연도", "상권ID", CODE]
    features = [*FEATURES, *POPULATION_FEATURES]
    if data.duplicated(key).any() or data[[*FEATURES, TARGET, "split"]].isna().any().any():
        raise ValueError("중복 행 또는 필수 경쟁 특성·목표값 결측이 있습니다.")
    if not np.isfinite(data[[*FEATURES[1:], TARGET]].to_numpy(dtype=float)).all() or data[TARGET].lt(0).any():
        raise ValueError("유효하지 않은 수치 특성 또는 목표값이 있습니다.")

    columns = [*key, *[name for name in features if name != CODE], TARGET, "split"]
    sorted_data = data.sort_values(key)
    output.mkdir(parents=True, exist_ok=True)
    sorted_data[columns].to_csv(output / "annual_per_store_training.csv", index=False, encoding="utf-8-sig")
    sorted_data[[*key, DONG, MONTH, "추정매출4분기합", COUNT, "포함분기수", "관측시간수",
                 "월_달력시간수", "유동인구_원본행정동코드", "유동인구_연결"]].to_csv(
        output / "annual_training_audit.csv", index=False, encoding="utf-8-sig"
    )
    population.to_csv(output / "dong_population_features.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset": "annual_per_store_training.csv", "target": TARGET,
        "target_kind": "annual_area_industry_sales_divided_by_one_observed_store_count",
        "identifier_columns": key, "feature_columns": features, "categorical_features": [CODE],
        "numeric_features": [name for name in features if name != CODE],
        "train_years": sorted(data.loc[data["split"].eq("train"), "기준연도"].unique().tolist()),
        "test_years": sorted(data.loc[data["split"].eq("test"), "기준연도"].unique().tolist()),
        "split_counts": data["split"].value_counts().to_dict(),
        "population_matches_by_split": data.groupby("split")["유동인구_연결"].sum().to_dict(),
        "population_missing_by_feature": data[POPULATION_FEATURES].isna().sum().to_dict(),
        "test_rows_unseen_industry": int((data["split"].eq("test") & ~data[CODE].isin(
            data.loc[data["split"].eq("train"), CODE]
        )).sum()),
        "population_months": sorted(set(population[MONTH])), "serving_month": serving_month,
    }
    manifest = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (output / "annual_training_manifest.json").write_text(manifest, encoding="utf-8")
    print(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--population", type=Path)
    parser.add_argument("--serving-month", default="202512")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.project / "data/processed/training"
    population = args.population or folder.parent / "안양시_유동인구_2023_2025.csv"
    prepare(folder, args.output or folder, population, args.boundary, args.serving_month)


if __name__ == "__main__":
    main()
