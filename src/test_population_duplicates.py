import csv
from pathlib import Path
from collections import Counter


def count_duplicate_keys(rows):
    counts = Counter(rows)
    return sum(count > 1 for count in counts.values())


# 각 행의 순서: 날짜, 행정동 코드, 시간 코드, 내외국인 구분
domestic = ("20230104", "41173570", "4", "내국인")
foreign = ("20230104", "41173570", "4", "외국인")

test_cases = [
    ("네 값이 모두 같음", [domestic, domestic], 1),
    ("내외국인 구분만 다름", [domestic, foreign], 0),
]

for name, rows, expected in test_cases:
    actual = count_duplicate_keys(rows)
    print(f"{name} | 기대 {expected}개 | 실제 {actual}개")
    assert actual == expected, f"테스트 실패: {name}"

print("테스트 2개 통과")

# --------------------------------- 실제 데이터 접근 -------------------------
project = Path(__file__).resolve().parents[1]
filename = "유동인구_행정동 단위 집계_202301_4117_안양시.csv"
matches = list((project / "data" / "raw").rglob(filename))

if len(matches) != 1:
    raise ValueError(f"1월 원본이 1개 있어야 합니다. 찾은 파일: {len(matches)}개")

key_columns = ("ETL_YMD", "ADMI_CD", "TIME_CD", "FORN_GB")

with matches[0].open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    missing = [column for column in key_columns if column not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    keys = [tuple(row[column] for column in key_columns) for row in reader]

print(f"전체 행: {len(keys):,}개")
print(f"중복 조합: {count_duplicate_keys(keys):,}개")