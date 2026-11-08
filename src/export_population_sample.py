from pathlib import Path
import json
import pandas as pd

# 실행 위치와 관계없이 프로젝트 폴더 기준
root = Path(__file__).resolve().parents[1]
source = root / "data/processed/안양시_유동인구_2023_2025.csv"
output = root / "results/03_분석준비"
output.mkdir(parents=True, exist_ok=True)

if not source.is_file():
    raise FileNotFoundError(f"통합 CSV가 없습니다: {source}")

samples = []
unique_values = {}
missing = {}
row_count = 0
columns = None

# 원본은 읽기만 하며, 코드의 앞자리 0을 유지
for chunk in pd.read_csv(
    source,
    encoding="utf-8-sig",
    dtype="string",
    chunksize=50_000,
):
    if columns is None:
        columns = chunk.columns.tolist()
        missing = {col: 0 for col in columns}
        unique_values = {
            col: set()
            for col in [
                "ADMI_CD", "TIME_CD", "FORN_GB", "기준연월"
            ]
            if col in columns
        }

    row_count += len(chunk)

    for col in columns:
        missing[col] += int(chunk[col].isna().sum())

    for col in unique_values:
        unique_values[col].update(chunk[col].dropna().tolist())

    # 파일 전체 구간에서 일부 행 추출 — 통계 분석용 표본은 아님
    samples.append(
        chunk.sample(n=min(10, len(chunk)), random_state=42)
    )

sample = pd.concat(samples, ignore_index=True)
sample.to_csv(
    output / "유동인구_확인용샘플.csv",
    index=False,
    encoding="utf-8-sig",
)

report = {
    "원본파일": source.name,
    "전체행수": row_count,
    "컬럼수": len(columns),
    "실제컬럼": columns,
    "컬럼별결측수": missing,
    "주요코드실제값": {
        col: sorted(values)
        for col, values in unique_values.items()
    },
    "주의": "샘플은 구조 확인용이며 집계 검증이나 대표성 있는 분석용이 아님",
}

report_path = output / "유동인구_구조확인.json"
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("완료! 아래 폴더의 두 파일을 보내주세요.")
print(output)