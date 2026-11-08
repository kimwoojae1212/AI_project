import pandas as pd
from pathlib import Path
import re

data_folder = Path("data/raw")
output = Path("results/02_데이터품질검사")
output.mkdir(parents=True, exist_ok=True)

# 기존 파일에서 안양시 지역코드 찾기
all_files = list(data_folder.rglob("*.csv"))
codes = set()

for file in all_files:
    if "유동인구" not in file.name:
        continue

    match = re.search(r"_(\d{4,5})_안양시$", file.stem)
    if match:
        codes.add(match.group(1))

# 안양시 파일 36개 검색
files = []

for file in all_files:
    if "유동인구" not in file.name:
        continue

    match = re.search(r"_(\d{4,5})$", file.stem)

    if "안양시" in file.name or (
        match and match.group(1) in codes
    ):
        files.append(file)

files = sorted(set(files))
print("검사 대상:", len(files), "개")

results = []
value_results = []
key_results = []

for i, file in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {file.name}")

    try:
        df = None

        for enc in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("인코딩 확인 실패")

        df.columns = df.columns.str.strip()

        # 주요 코드 컬럼 값 확인
        for col in ["ADMI_CD", "TIME_CD", "FORN_GB"]:
            if col in df.columns:
                values = df[col].value_counts(
                    dropna=False
                )

                for value, count in values.items():
                    value_results.append({
                        "파일명": file.name,
                        "컬럼": col,
                        "값": str(value),
                        "건수": int(count)
                    })

        # 인구 관련 컬럼 검사
        pop_cols = [
            col for col in df.columns
            if re.match(r"^[MFW]_\d+_CNT$", col)
        ]

        negative_count = 0
        non_numeric_count = 0

        for col in pop_cols:
            numeric = pd.to_numeric(
                df[col], errors="coerce"
            )

            negative_count += int((numeric < 0).sum())

            non_numeric_count += int(
                (numeric.isna() & df[col].notna()).sum()
            )

        # 집계 키 중복 여부
        # 실제 고유 키는 아직 확정되지 않았음
        candidate_keys = [
            "ADMI_CD", "TIME_CD", "FORN_GB"
        ]

        available_keys = [
            col for col in candidate_keys
            if col in df.columns
        ]

        duplicate_keys = None

        if len(available_keys) == len(candidate_keys):
            duplicate_keys = int(
                df.duplicated(
                    subset=available_keys
                ).sum()
            )

        key_results.append({
            "파일명": file.name,
            "검사키": ", ".join(available_keys),
            "후보키중복행수": duplicate_keys
        })

        results.append({
            "파일명": file.name,
            "행수": len(df),
            "행정동수": (
                df["ADMI_CD"].nunique()
                if "ADMI_CD" in df.columns else None
            ),
            "인구컬럼수": len(pop_cols),
            "음수값수": negative_count,
            "숫자변환실패수": non_numeric_count,
            "ETL_YMD_고유값수": (
                df["ETL_YMD"].nunique()
                if "ETL_YMD" in df.columns else None
            ),
            "상태": "성공"
        })

    except Exception as e:
        results.append({
            "파일명": file.name,
            "상태": f"오류: {e}"
        })

# 결과 저장
pd.DataFrame(results).to_csv(
    output / "안양시_유동인구_값검사.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(value_results).to_csv(
    output / "안양시_유동인구_코드값분포.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(key_results).to_csv(
    output / "안양시_유동인구_집계키검사.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n검사 완료!")