from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Current": BASE_DIR / "datasets" / "data.csv",
    "114K": BASE_DIR / "datasets" / "candidate_datasets" / "114k" / "spotify-tracks-dataset-detailed.csv",
    "900K": BASE_DIR / "datasets" / "candidate_datasets" / "900k" / "tracks.csv",
}

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "column_inspection_report.txt"


def write(text=""):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(str(text))
        f.write("\n")


def main():

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    datasets = {}

    for name, path in DATASETS.items():
        df = pd.read_csv(path, nrows=5, low_memory=False)
        datasets[name] = set(df.columns)

        write("=" * 80)
        write(name)
        write("=" * 80)

        for col in sorted(df.columns):
            write(col)

        write()

    write("=" * 80)
    write("COMMON COLUMNS")
    write("=" * 80)

    common = set.intersection(*datasets.values())

    for col in sorted(common):
        write(col)

    write()

    for dataset_name, columns in datasets.items():

        write("=" * 80)
        write(f"UNIQUE COLUMNS : {dataset_name}")
        write("=" * 80)

        unique = columns - common
        for col in sorted(unique):
            write(col)

        write()


if __name__ == "__main__":
    main()