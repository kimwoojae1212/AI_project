"""판매 CSV로 Random Forest 회귀 모델을 학습하고 평가 자료를 저장한다."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
PILOT_NAME = "anyang_sales_training_pilot.csv"
MODEL_NAME = "random_forest"
RANDOM_STATE = 42
PILOT_EXCLUSIONS = (
    "매출건수", "분기말_관측점포수", "점포수_기준연월", "점포당_분기매출_참고값"
)
OUTPUT_COLUMNS = ("y_true", "y_pred", "absolute_error")


def load_data(path, target_col, exclude_cols, encoding):
    raw = pd.read_csv(
        path, encoding=encoding, low_memory=False,
        na_values=["inf", "+inf", "-inf", "Infinity", "+Infinity", "-Infinity"],
    )
    target_col = target_col or ("매출금액" if "매출금액" in raw else raw.columns[-1])
    if target_col not in raw:
        raise ValueError(f"정답 열을 찾을 수 없습니다. 지정한 열: {target_col}")

    if exclude_cols is None:
        exclude_cols = []
        if path.name == PILOT_NAME and target_col == "매출금액":
            exclude_cols = [col for col in PILOT_EXCLUSIONS if col in raw]

    unknown = set(exclude_cols).difference(raw.columns)
    if unknown or target_col in exclude_cols:
        raise ValueError(f"제외 열을 확인하세요. 지정한 열: {exclude_cols}")
    X = raw.drop(columns=[target_col, *exclude_cols])
    if X.shape[1] == 0 or set(OUTPUT_COLUMNS).intersection(X.columns):
        raise ValueError(f"입력 열이 비어 있거나 결과 열 이름과 겹칩니다. 결과 열: {OUTPUT_COLUMNS}")

    # 실제 정답이 없는 행은 회귀 학습과 평가에서 제외한다.
    raw_target = raw[target_col]
    y = pd.to_numeric(raw_target.astype(str).str.replace(",", "", regex=False), errors="coerce")
    invalid = raw_target.notna() & y.isna()
    if invalid.any():
        raise ValueError(f"정답 열 {target_col}에 숫자로 변환할 수 없는 값이 {invalid.sum():,}개 있습니다.")
    valid = np.isfinite(y)
    if valid.sum() < 6:
        raise ValueError("80:20 분할 후 양쪽의 R²를 계산하려면 유효한 정답이 6개 이상 필요합니다.")
    print(f"정답 열: {target_col} | 정답 결측 또는 무한값 제외: {(~valid).sum():,}행")
    print(f"제외한 입력 열: {', '.join(exclude_cols) if exclude_cols else '없음'}")
    return X.loc[valid], y.loc[valid], target_col, exclude_cols


def build_pipeline(X_train):
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.columns.difference(numeric, sort=False).tolist()
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        # 원-핫 인코더는 결측값을 별도의 범주로 취급한다.
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return pipeline, numeric, categorical


def evaluate(y, predicted):
    truth = y.to_numpy(dtype=float)
    error = np.abs(truth - predicted)
    nonzero = truth != 0
    return {
        "r2": float(r2_score(truth, predicted)) if np.ptp(truth) > 0 else None,
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "mape_pct": float(100 * np.mean(error[nonzero] / np.abs(truth[nonzero]))) if nonzero.any() else None,
        "mape_excluded_zero_count": int((~nonzero).sum()),
    }


def save_importance(pipeline, numeric, categorical, csv_path, png_path):
    preprocess = pipeline.named_steps["preprocess"]
    weights = pipeline.named_steps["model"].feature_importances_
    totals = list(weights[preprocess.output_indices_["numeric"]])
    if categorical:
        encoder = preprocess.named_transformers_["categorical"]
        encoded_weights = weights[preprocess.output_indices_["categorical"]]
        offset = 0
        # 같은 원본 열에서 만들어진 원-핫 특성의 중요도를 합산한다.
        for levels in encoder.categories_:
            end = offset + len(levels)
            totals.append(encoded_weights[offset:end].sum())
            offset = end

    importance = pd.DataFrame({"feature": numeric + categorical, "importance": totals})
    importance = importance.sort_values("importance", ascending=False, kind="stable")
    importance.to_csv(csv_path, index=False, encoding="utf-8-sig")

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
        if family in installed:
            plt.rcParams["font.family"] = family
            break
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(top) + 1.5)), layout="constrained")
    ax.barh(top["feature"], top["importance"], color="#28677D")
    ax.set(title="Random Forest | Top 10 Features", xlabel="Impurity-based importance")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def train(args):
    print(f"데이터 불러오는 중: {args.input}")
    X, y, target_col, excluded = load_data(args.input, args.target_col, args.exclude_cols, args.encoding)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"데이터 분할 완료 | 학습 {len(X_train):,}행 | 테스트 {len(X_test):,}행")
    pipeline, numeric, categorical = build_pipeline(X_train)
    print("Random Forest 학습 중")
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    test_pred = pipeline.predict(X_test)
    inference_seconds = perf_counter() - start
    train_pred = pipeline.predict(X_train)

    metrics = {
        "model": MODEL_NAME,
        "input_file": args.input.name,
        "target": target_col,
        "excluded_feature_columns": excluded,
        "random_state": RANDOM_STATE,
        "test_fraction": 0.2,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_estimators": pipeline.named_steps["model"].n_estimators,
        "fit_seconds": fit_seconds,
        "test_inference_seconds": inference_seconds,
        "train": evaluate(y_train, train_pred),
        "test": evaluate(y_test, test_pred),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_dir / MODEL_NAME
    Path(f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions = X_test.assign(
        y_true=y_test.to_numpy(), y_pred=test_pred, absolute_error=np.abs(y_test.to_numpy() - test_pred)
    )
    predictions.sort_index().to_csv(f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    save_importance(pipeline, numeric, categorical, f"{prefix}_feature_importance.csv", f"{prefix}_feature_importance.png")
    joblib.dump(pipeline, f"{prefix}_model.pkl", compress=3)
    print(f"테스트 평가: {metrics['test']}")
    print(f"결과 5개 저장 완료: {args.results_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = ROOT / "data" / PILOT_NAME
    processed_input = ROOT / "data" / "processed" / PILOT_NAME
    if not default_input.is_file() and processed_input.is_file():
        default_input = processed_input
    parser.add_argument("--input", type=Path, default=default_input, help="입력 CSV 경로")
    parser.add_argument("--target-col", help="정답 열 이름. 생략하면 매출금액 또는 마지막 열")
    parser.add_argument("--exclude-cols", nargs="*", help="예측에서 제외할 입력 열 이름 목록")
    parser.add_argument("--encoding", default="utf-8-sig", help="입력 CSV 인코딩")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="결과 저장 폴더")
    try:
        train(parser.parse_args())
    except (OSError, ValueError) as exc:
        parser.exit(1, f"실행 오류: {exc}\n")


if __name__ == "__main__":
    main()
