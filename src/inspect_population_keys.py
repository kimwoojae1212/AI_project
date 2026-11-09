import argparse
import csv
from collections import Counter
from pathlib import Path

KEY_COLUMNS = ("ETL_YMD", "ADMI_CD", "TIME_CD", "FORN_GB")

parser = argparse.ArgumentParser(description="유동인구 CSV의 중복 조합을 검사합니다.")
parser.add_argument("csv_path", type=Path, help="검사할 CSV 경로")
args = parser.parse_args()

with args.csv_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    missing = [col for col in KEY_COLUMNS if col not in (reader.fieldnames or [])]
    if missing:
        parser.error(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    counts = Counter(tuple(row[col] for col in KEY_COLUMNS) for row in reader)

print(f"파일: {args.csv_path.name}")
print(f"전체 행: {sum(counts.values()):,}개")
print(f"중복 조합: {sum(count > 1 for count in counts.values()):,}개")