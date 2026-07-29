from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Current": BASE_DIR / "datasets" / "data.csv",
    "114K": BASE_DIR / "datasets" / "candidate_datasets" / "114k" / "spotify-tracks-dataset-detailed.csv",
    "900K": BASE_DIR / "datasets" / "candidate_datasets" / "900k" / "tracks.csv",
}

REPORT_PATH = BASE_DIR / "reports"
REPORT_PATH.mkdir(exist_ok=True)

REPORT_FILE = REPORT_PATH / "dataset_comparison_report.txt"

REQUIRED_COLUMNS = [
    "id",
    "name",
    "artists",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
]


def write(text="", end="\n"):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(str(text) + end)


def analyze_dataset(name: str, path: Path):

    write("=" * 80)
    write(name)
    write("=" * 80)

    df = pd.read_csv(path, low_memory=False)

    write(f"Shape               : {df.shape}")
    write(
        f"Memory              : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
    )

    write("\nColumns")
    write("-" * 80)

    for col in df.columns:
        write(col)

    write("\nMissing Values")
    write("-" * 80)

    write(df.isna().sum().sort_values(ascending=False).head(15).to_string())

    write("\nDuplicate Rows")
    write("-" * 80)

    write(df.duplicated().sum())

    write("\nRequired Columns")
    write("-" * 80)

    for col in REQUIRED_COLUMNS:
        write(f"{col:<20} {'YES' if col in df.columns else 'NO'}")

    write("\nRelease Year")

    if "year" in df.columns:
        write(f"{df['year'].min()} -> {df['year'].max()}")

    elif "release_date" in df.columns:

        years = pd.to_datetime(
            df["release_date"],errors="coerce"
        ).dt.year

        write(f"{years.min()} -> {years.max()}")

    elif "album_release_date" in df.columns:

        years = pd.to_datetime(
            df["album_release_date"],errors="coerce"
        ).dt.year

        write(f"{years.min()} -> {years.max()}")

    else:
        write("No year information found.")

    write("\nFirst Five Rows")
    write("-" * 80)

    write(df.head().to_string())

    write("\n\n")


def main():

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    for name, path in DATASETS.items():
        analyze_dataset(name, path)

    print(f"\nReport saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()