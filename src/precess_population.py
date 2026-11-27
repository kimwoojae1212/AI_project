import pandas as pd
from pathlib import Path
import re

# 경로 설정
data_folder = Path("data/raw")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# 전체 CSV 검색
all_files = list(data_folder.rglob("*.csv"))

# 1. 기존 파일에서 안양시 지역코드 추출
codes = set()

for file in all_files:
    if "유동인구" not in file.name:
        continue

    match = re.search(r"_(\d{4,5})_안양시$", file.stem)

    if match:
        codes.add(match.group(1))

# 2. 안양시 유동인구 파일 검색
files = []

for file in all_files:
    if "유동인구" not in file.name:
        continue

    code_match = re.search(r"_(\d{4,5})$", file.stem)

    if "안양시" in file.name or (
        code_match and code_match.group(1) in codes
    ):
        files.append(file)

files = sorted(set(files))

print("통합 대상 파일:", len(files))

if len(files) != 36:
    raise ValueError("36개 파일이 아닙니다. 파일 목록을 확인하세요.")

# 3. 파일별 데이터 읽기
dataframes = []

for i, file in enumerate(files, 1):

    print(f"[{i}/{len(files)}] {file.name}")

    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(
                file,
                encoding=enc,
                low_memory=False,
                dtype={"ADMI_CD": "string", "FORN_GB": "string"}
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"읽기 실패: {file.name}")

    df.columns = df.columns.str.strip()

    # 4. 내국인·외국인 표기 통일
    if "FORN_GB" in df.columns:
        df["FORN_GB"] = (
            df["FORN_GB"]
            .str.strip()
            .replace({
                "N": "내국인",
                "F": "외국인"
            })
        )

        allowed = {"내국인", "외국인"}
        unknown = set(df["FORN_GB"].dropna().unique()) - allowed

        if unknown:
            raise ValueError(
                f"{file.name}: 알 수 없는 FORN_GB 값 {unknown}"
            )

    # 5. 날짜 변환
    if "ETL_YMD" in df.columns:
        df["ETL_YMD"] = pd.to_datetime(
            df["ETL_YMD"].astype("string"),
            format="%Y%m%d",
            errors="raise"
        )

    # 6. 파일명에서 기준 연월 추출
    match = re.search(r"(20\d{2})(0[1-9]|1[0-2])", file.stem)

    if not match:
        raise ValueError(f"연월 확인 실패: {file.name}")

    df["기준연월"] = match.group(0)
    df["원본파일"] = file.name

    dataframes.append(df)

# 7. 전체 데이터 통합
combined = pd.concat(
    dataframes,
    ignore_index=True
)

# 8. 통합 후 중복 검사
keys = ["ETL_YMD", "ADMI_CD", "TIME_CD", "FORN_GB"]

duplicate_count = combined.duplicated(
    subset=keys
).sum()

print("\n===== 통합 결과 =====")
print("전체 행 수:", len(combined))
print("전체 열 수:", len(combined.columns))
print("후보 키 중복:", duplicate_count)

if duplicate_count > 0:
    raise ValueError("통합 후 후보 키 중복이 발견됐습니다.")

# 9. 날짜순 정렬
combined = combined.sort_values(
    by=["ETL_YMD", "ADMI_CD", "TIME_CD"]
)

# 10. CSV 저장
output_path = output_folder / "안양시_유동인구_2023_2025.csv"

combined.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n통합 완료!")
print("저장 위치:", output_path)