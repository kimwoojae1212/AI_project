import pandas as pd
from pathlib import Path

# 경로 설정
data_folder = Path("data/raw")
output_folder = Path("results/02_데이터품질검사")
output_folder.mkdir(parents=True, exist_ok=True)

# 안양시 유동인구 파일 검색
# 기존 안양시 파일 35개
files = [
    file for file in data_folder.rglob("*.csv")
    if "유동인구" in file.name and "안양시" in file.name
]

# 기존 파일명에서 안양시 지역코드 찾기
import re

codes = set()

for file in files:
    match = re.search(r"_(\d{4,5})_안양시$", file.stem)
    if match:
        codes.add(match.group(1))

print("안양시 지역코드:", codes)

# 지역코드만 있는 파일 추가
for file in data_folder.rglob("*.csv"):
    if "유동인구" not in file.name:
        continue

    match = re.search(r"_(\d{4,5})$", file.stem)

    if match and match.group(1) in codes:
        files.append(file)

files = sorted(set(files))
print(f"안양시 유동인구 파일: {len(files)}개")

results = []
column_sets = {}

for i, file in enumerate(files, 1):

    print(f"[{i}/{len(files)}] {file.name}")

    try:
        # 인코딩 자동 시도
        df = None

        for enc in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("인코딩 확인 실패")

        # 컬럼명 앞뒤 공백 제거
        df.columns = df.columns.str.strip()

        # 컬럼 구조 기록
        column_sets[file.name] = tuple(df.columns)

        # 데이터 품질 검사
        results.append({
            "파일명": file.name,
            "행수": len(df),
            "열수": len(df.columns),
            "결측치수": int(df.isna().sum().sum()),
            "결측치포함행수": int(df.isna().any(axis=1).sum()),
            "중복행수": int(df.duplicated().sum()),
            "인코딩": enc,
            "상태": "성공"
        })

    except Exception as e:
        results.append({
            "파일명": file.name,
            "행수": None,
            "열수": None,
            "결측치수": None,
            "결측치포함행수": None,
            "중복행수": None,
            "인코딩": None,
            "상태": str(e)
        })

# 검사 결과 저장
result_df = pd.DataFrame(results)

result_df.to_csv(
    output_folder / "안양시_유동인구_품질검사.csv",
    index=False,
    encoding="utf-8-sig"
)

# 컬럼 구조 비교
if column_sets:
    standard = next(iter(column_sets.values()))

    column_results = []

    for filename, columns in column_sets.items():
        column_results.append({
            "파일명": filename,
            "컬럼수": len(columns),
            "기준컬럼과일치": columns == standard,
            "컬럼명": ", ".join(columns)
        })

    pd.DataFrame(column_results).to_csv(
        output_folder / "안양시_유동인구_컬럼비교.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n===== 검사 결과 =====")
print(result_df.to_string(index=False))

print("\n검사 완료!")