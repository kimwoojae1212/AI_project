import pandas as pd
from pathlib import Path
import re

data_folder = Path("data/raw")
output = Path("results/02_데이터품질검사")
output.mkdir(parents=True, exist_ok=True)

# 1. 안양시 지역코드 확인
all_files = list(data_folder.rglob("*.csv"))
codes = set()

for file in all_files:
    if "유동인구" not in file.name:
        continue

    match = re.search(r"_(\d{4,5})_안양시$", file.stem)

    if match:
        codes.add(match.group(1))

# 2. 안양시 파일 검색
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
print("검사 파일 수:", len(files))

data = {}

for file in files:
    date_match = re.search(
        r"(20\d{2})(0[1-9]|1[0-2])",
        file.stem
    )

    if not date_match:
        continue

    ym = date_match.group(0)

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
        print("읽기 실패:", file.name)
        continue

    df.columns = df.columns.str.strip()
    data[ym] = df

# 3. 2025년 5월 행정동 비교
target = data["202505"]

other_codes = set()

for ym, df in data.items():
    if ym != "202505":
        other_codes.update(
            df["ADMI_CD"].dropna().astype(str)
        )

may_codes = set(
    target["ADMI_CD"].dropna().astype(str)
)

missing_codes = sorted(other_codes - may_codes)

print("\n===== 2025년 5월에 없는 행정동 코드 =====")
print(missing_codes)

pd.DataFrame({
    "2025년5월_미등장행정동코드": missing_codes
}).to_csv(
    output / "202505_행정동차이.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. ETL_YMD 값 확인
date_results = []

for ym, df in data.items():
    if "ETL_YMD" in df.columns:
        date_results.append({
            "연월": ym,
            "ETL_YMD_고유값수": df["ETL_YMD"].nunique(),
            "최솟값": str(df["ETL_YMD"].min()),
            "최댓값": str(df["ETL_YMD"].max())
        })

pd.DataFrame(date_results).to_csv(
    output / "ETL_YMD_검사.csv",
    index=False,
    encoding="utf-8-sig"
)

# 5. 후보 고유 키 검사
key_results = []

candidates = [
    ["ADMI_CD", "TIME_CD", "FORN_GB"],
    ["ETL_YMD", "ADMI_CD", "TIME_CD", "FORN_GB"]
]

for ym, df in data.items():

    for keys in candidates:
        if all(col in df.columns for col in keys):
            duplicate_count = df.duplicated(
                subset=keys
            ).sum()

            key_results.append({
                "연월": ym,
                "후보키": " + ".join(keys),
                "중복행수": int(duplicate_count),
                "전체행수": len(df)
            })

pd.DataFrame(key_results).to_csv(
    output / "안양시_후보키_재검사.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n검사 완료!")