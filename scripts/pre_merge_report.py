from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Current": BASE_DIR / "datasets" / "data.csv",
    "114K": BASE_DIR / "datasets" / "processed" / "114k_normalized.csv",
    "900K": BASE_DIR / "datasets" / "processed" / "900k_normalized.csv",
}

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT = REPORT_DIR / "pre_merge_report.txt"


def write(text=""):
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(str(text))
        f.write("\n")


if REPORT.exists():
    REPORT.unlink()

for name, path in DATASETS.items():

    df = pd.read_csv(path, low_memory=False)

    write("=" * 80)
    write(name)
    write("=" * 80)

    write(f"Rows : {len(df):,}")

    for col in ["id", "name", "artists", "release_date", "year"]:
        if col in df.columns:
            write(f"Missing {col:<12}: {df[col].isna().sum():,}")

    if "id" in df.columns:
        write(f"Duplicate IDs      : {df['id'].duplicated().sum():,}")

    if {"name","artists"}.issubset(df.columns):
        dup = df.duplicated(subset=["name","artists"]).sum()

        write(f"Duplicate Song+Artist : {dup:,}")

    eligible = df[
        df["id"].notna() & df["name"].notna() & df["artists"].notna()
        ]

    write(f"Eligible Rows : {len(eligible):,}")

    write()