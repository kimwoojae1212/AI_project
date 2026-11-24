"""팀 공통 무작위 분할로 연매출 랜덤포레스트를 학습한다."""

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "processed" / "training"
TARGET = "점포당_연매출_참고값"
MODEL_NAME = "randomforest"


def load_data(csv_path, manifest_path, target_col):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    features = manifest["feature_columns"]
    numeric = manifest["numeric_features"]
    categorical = manifest["categorical_features"]
    if len(features) != 9 or sorted(features) != sorted(numeric + categorical):
        raise ValueError("명세에 수치형·범주형 입력 9개를 중복 없이 지정해야 합니다.")
    if len(set(features)) != 9 or target_col in features:
        raise ValueError("입력 컬럼은 중복될 수 없으며 목표 컬럼과 달라야 합니다.")
    data = pd.read_csv(csv_path, dtype={column: object for column in categorical})
    inputs = data[features].copy()
    for column in numeric:
        inputs[column] = pd.to_numeric(inputs[column], errors="raise")
    target = pd.to_numeric(data[target_col], errors="raise")
    valid = np.isfinite(target.to_numpy(dtype=float, na_value=np.nan))
    if not valid.all():
        print(f"목표값이 없거나 유한하지 않은 {int((~valid).sum())}행을 제외합니다.")
    return data, inputs.loc[valid], target.loc[valid], numeric, categorical


def build_pipeline(numeric, categorical):
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", category_pipeline, categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])


def evaluate(pipeline, split_name, inputs, target, fit_seconds):
    start = perf_counter()
    predictions = pipeline.predict(inputs)
    inference_seconds = perf_counter() - start
    actual = target.to_numpy(dtype=float)
    nonzero = actual != 0
    mape = (100 * np.mean(np.abs((actual[nonzero] - predictions[nonzero])
                                 / actual[nonzero])) if nonzero.any() else np.nan)
    metrics = {
        "split": split_name,
        "rows": len(target),
        "r2": r2_score(target, predictions),
        "mae": mean_absolute_error(target, predictions),
        "rmse": np.sqrt(mean_squared_error(target, predictions)),
        "mape_pct": mape,
        "mape_zero_targets_excluded": int((~nonzero).sum()),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }
    return metrics, predictions


def feature_importance(pipeline, numeric, categorical):
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    values = pipeline.named_steps["regressor"].feature_importances_
    # 인코딩 범주별 중요도를 원본 변수 단위로 합산한다.
    by_feature = dict(zip(numeric, values[preprocessor.output_indices_["numeric"]]))
    offset = preprocessor.output_indices_["categorical"].start
    for column, categories in zip(categorical, encoder.categories_):
        end = offset + len(categories)
        by_feature[column] = values[offset:end].sum()
        offset = end
    return (pd.Series(by_feature, name="importance").rename_axis("feature")
            .reset_index().sort_values("importance", ascending=False))


def korean_font(font_path):
    if font_path is not None:
        return font_manager.FontProperties(fname=font_path)
    for family in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return font_manager.FontProperties(fname=path)
    raise ValueError("한글 글꼴을 설치하거나 --font-path로 글꼴 파일을 지정하세요.")


def save_importance_chart(importance, path, font):
    top = importance.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    axis.barh(top["feature"], top["importance"], color="#286a8c")
    for label in axis.get_yticklabels():
        label.set_fontproperties(font)
    axis.set_title(f"랜덤포레스트 변수 중요도 상위 {len(top)}개", fontproperties=font)
    axis.set_xlabel("불순도 감소 기반 중요도", fontproperties=font)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=TRAINING_DIR / "annual_per_store_training.csv")
    parser.add_argument("--manifest", type=Path, default=TRAINING_DIR / "annual_training_manifest.json")
    parser.add_argument("--target-col", default=TARGET)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()

    font = korean_font(args.font_path)
    print(f"연매출 학습 데이터 읽는 중: {args.csv}")
    data, inputs, target, numeric, categorical = load_data(args.csv, args.manifest, args.target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        inputs, target, test_size=0.2, random_state=42,
    )
    print(f"무작위 80:20 분할 완료: 학습 {len(x_train)}행, 테스트 {len(x_test)}행")
    print("학습 세트로 결측치 처리와 인코딩을 맞추고 랜덤포레스트를 학습합니다.")
    pipeline = build_pipeline(numeric, categorical)
    start = perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    train_metrics, _ = evaluate(pipeline, "train", x_train, y_train, fit_seconds)
    test_metrics, predicted = evaluate(pipeline, "test", x_test, y_test, fit_seconds)

    # 원본 비교표에는 행 식별용 컬럼과 전처리 전 입력값을 함께 남긴다.
    comparison = data.loc[x_test.index].drop(columns=args.target_col).copy()
    comparison["y_true"] = y_test
    comparison["y_pred"] = predicted
    comparison["absolute_error"] = np.abs(y_test.to_numpy() - predicted)
    importance = feature_importance(pipeline, numeric, categorical)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_dir / MODEL_NAME
    pd.DataFrame([train_metrics, test_metrics]).to_csv(
        f"{prefix}_metrics.csv", index=False, encoding="utf-8-sig",
    )
    comparison.to_csv(f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(f"{prefix}_feature_importance.csv", index=False, encoding="utf-8-sig")
    save_importance_chart(importance, f"{prefix}_feature_importance.png", font)
    with open(f"{prefix}_model.pkl", "wb") as model_file:
        pickle.dump(pipeline, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"학습 R² {train_metrics['r2']:.4f}, 테스트 R² {test_metrics['r2']:.4f}")
    print(f"결과 파일 5개 저장 완료: {args.results_dir}")


if __name__ == "__main__":
    main()
