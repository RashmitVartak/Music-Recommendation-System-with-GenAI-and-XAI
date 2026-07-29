from pathlib import Path
import pandas as pd

from column_mapper import (DATASET_114K_MAPPING,DATASET_900K_MAPPING,CANONICAL_COLUMNS,)

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets"

INPUT_FILES = {
    "114K": DATASET_DIR / "candidate_datasets" / "114k" / "spotify-tracks-dataset-detailed.csv",
    "900K": DATASET_DIR / "candidate_datasets" / "900k" / "tracks.csv",
}

OUTPUT_DIR = DATASET_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "normalization_report.txt"


def write(text=""):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(str(text))
        f.write("\n")


def normalize_dataset(dataset_name: str, file_path: Path):

    write("=" * 80)
    write(dataset_name)
    write("=" * 80)

    print(f"Loading {dataset_name}...")

    df = pd.read_csv(file_path, low_memory=False)
    original_shape = df.shape

    # Rename columns
    if dataset_name == "114K":
        df.rename(columns=DATASET_114K_MAPPING, inplace=True)

    elif dataset_name == "900K":
        df.rename(columns=DATASET_900K_MAPPING, inplace=True)

    # Create year column
    if "year" not in df.columns:
        if "release_date" in df.columns:
            df["year"] = (
                pd.to_datetime(df["release_date"],errors="coerce").dt.year
            )

        else:
            df["year"] = pd.NA

    # Add missing canonical columns
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    # Keep only canonical columns
    
    df = df[CANONICAL_COLUMNS]
    output_file = OUTPUT_DIR / f"{dataset_name.lower()}_normalized.csv"
    df.to_csv(output_file, index=False)

    write(f"Original Shape : {original_shape}")
    write(f"Normalized Shape : {df.shape}")
    write(f"Saved To : {output_file}")
    write()

    print(f"Saved {dataset_name}.")


def main():

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    for dataset_name, path in INPUT_FILES.items():
        normalize_dataset(dataset_name, path)

    print("\nNormalization Complete.")
    print(f"Report saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()