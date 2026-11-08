import pandas as pd
from pathlib import Path
import re
import calendar

# 경로
data_folder = Path("data/raw")
output = Path("results/02_데이터품질검사")
output.mkdir(parents=True, exist_ok=True)

all_files = list(data_folder.rglob("*.csv"))

# 안양시 지역코드 추출
codes = set()

for file in all_files:
    if "유동인구" not in file.name:
        continue

    match = re.search(r"_(\d{4,5})_안양시$", file.stem)

    if match:
        codes.add(match.group(1))

# 필요한 4개월만 검색
target_months = {
    "202303", "202504", "202505", "202506"
}

files = []

for file in all_files:
    if "유동인구" not in file.name:
        continue

    date_match = re.search(r"20\d{4}", file.stem)

    if not date_match:
        continue

    ym = date_match.group()

    if ym not in target_months:
        continue

    code_match = re.search(r"_(\d{4,5})$", file.stem)

    if "안양시" in file.name or (
        code_match and code_match.group(1) in codes
    ):
        files.append((ym, file))

# 데이터 읽기
data = {}

for ym, file in files:
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(
                file,
                encoding=enc,
                low_memory=False
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"읽기 실패: {file.name}")

    df.columns = df.columns.str.strip()
    data[ym] = df

# 필수 파일 확인
missing_files = target_months - set(data)

if missing_files:
    raise ValueError(
        f"필요한 파일 없음: {sorted(missing_files)}"
    )

# ==================================
# 1. 2023년 3월 누락 날짜 확인
# ==================================

df = data["202303"]

dates = pd.to_datetime(
    df["ETL_YMD"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

expected_dates = pd.date_range(
    "2023-03-01",
    "2023-03-31",
    freq="D"
)

actual_dates = pd.DatetimeIndex(
    dates.dropna().unique()
)

missing_dates = expected_dates.difference(actual_dates)

date_result = pd.DataFrame({
    "누락날짜": missing_dates.strftime("%Y-%m-%d")
})

date_result.to_csv(
    output / "202303_누락날짜.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 2023년 3월 누락 날짜 =====")
print(missing_dates.strftime("%Y-%m-%d").tolist())

print("날짜 변환 실패 행:", dates.isna().sum())

# ==================================
# 2. 2025년 4~6월 행정동 비교
# ==================================

month_codes = {}

for ym in ["202504", "202505", "202506"]:
    month_codes[ym] = set(
        data[ym]["ADMI_CD"]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

all_codes = sorted(set.union(*month_codes.values()))

comparison = []

for code in all_codes:
    comparison.append({
        "행정동코드": code,
        "2025년4월": code in month_codes["202504"],
        "2025년5월": code in month_codes["202505"],
        "2025년6월": code in month_codes["202506"]
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    output / "2025_행정동변화비교.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 월별 행정동 수 =====")

for ym, codes_set in month_codes.items():
    print(ym, len(codes_set))

print("\n===== 5월에만 없는 코드 =====")

missing_may = (
    month_codes["202504"]
    & month_codes["202506"]
) - month_codes["202505"]

print(sorted(missing_may))

print("\n===== 4월에서 5월 사이 사라진 코드 =====")
print(sorted(
    month_codes["202504"] - month_codes["202505"]
))

print("\n===== 4월에서 5월 사이 새로 등장한 코드 =====")
print(sorted(
    month_codes["202505"] - month_codes["202504"]
))

print("\n최종 검사 완료!")