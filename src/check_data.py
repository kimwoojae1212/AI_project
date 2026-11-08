import pandas as pd
from pathlib import Path

data_folder = Path("data/raw")
results = []

files = list(data_folder.rglob("*.csv"))

print(f"전체 CSV 파일: {len(files)}개")

for i, file in enumerate(files, 1):
    try:
        df = None

        for enc in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(
                    file,
                    encoding=enc,
                    nrows=0
                )
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("인코딩 확인 실패")

        results.append({
            "파일명": file.name,
            "경로": str(file),
            "컬럼수": len(df.columns),
            "컬럼명": ", ".join(df.columns),
            "인코딩": enc,
            "상태": "성공"
        })

    except Exception as e:
        results.append({
            "파일명": file.name,
            "경로": str(file),
            "컬럼수": None,
            "컬럼명": None,
            "인코딩": None,
            "상태": str(e)
        })

    print(f"[{i}/{len(files)}] {file.name}")

output = Path("results/01_데이터구조확인")
output.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    output / "전체파일요약.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n전체 데이터 구조 확인 완료!")