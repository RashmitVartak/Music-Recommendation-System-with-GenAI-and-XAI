from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "processed"

FILES = {
    "114K": DATASET_DIR / "114k_normalized.csv",
    "900K": DATASET_DIR / "900k_normalized.csv",
}

OUTPUT_DIR = DATASET_DIR
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "repair_report.txt"

DROP_ROWS_WITH_MISSING_ARTISTS_900K = True

# Report helper
def write(text=""):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(str(text))
        f.write("\n")

# Text cleaning
def clean_text(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value == "":
        return pd.NA

    return value

# Artist cleaning
def clean_artists(value):

    if pd.isna(value):
        return pd.NA

    value = str(value)

    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.replace("'", "")
    value = value.replace('"', "")

    value = value.replace(",", ";")

    value = ";".join(
        artist.strip()
        for artist in value.split(";")
        if artist.strip()
    )

    if value == "":
        return pd.NA

    return value

# Keep best duplicate
def completeness_score(row):

    score = 0
    important = [
        "release_date",
        "album_name",
        "genres",
        "artist_popularity",
        "artist_followers",
        "year",
    ]

    for col in important:
        if col in row.index and pd.notna(row[col]):
            score += 1

    return score

# Repair
def repair_dataset(dataset_name, path):

    write("=" * 80)
    write(dataset_name)
    write("=" * 80)

    print(f"Processing {dataset_name}...")

    df = pd.read_csv(path, low_memory=False)
    original_rows = len(df)

    # Clean text columns
    text_columns = [
        "id",
        "name",
        "artists",
        "album_name",
        "genres",
        "release_date",
    ]

    for col in text_columns:
        if col in df.columns:
            if col == "artists":
                df[col] = df[col].apply(clean_artists)
            else:
                df[col] = df[col].apply(clean_text)

    # Release Date
    if "release_date" in df.columns:

        df["release_date"] = pd.to_datetime(
            df["release_date"],
            errors="coerce"
        )

    # Year
    if "year" in df.columns:
        missing_year = (df["year"].isna() & df["release_date"].notna())
        df.loc[missing_year, "year"] = (df.loc[missing_year, "release_date"].dt.year)

    # Remove unusable rows
    before = len(df)
    df = df[ df["id"].notna() & df["name"].notna()]
    removed_missing_id_name = before - len(df)

    # 900K specific
    removed_missing_artists = 0

    if dataset_name == "900K" and DROP_ROWS_WITH_MISSING_ARTISTS_900K:
        before = len(df)
        df = df[df["artists"].notna()]
        removed_missing_artists = before - len(df)

    # Duplicate IDs
    duplicate_ids_before = df["id"].duplicated().sum()

    if duplicate_ids_before > 0:

        df["_score"] = df.apply(completeness_score,axis=1)
        df = (
            df
            .sort_values("_score", ascending=False)
            .drop_duplicates("id")
            .drop(columns="_score")
        )

    # Duplicate Songs
    duplicate_song_before = (df.duplicated(subset=["name", "artists"]).sum())

    if duplicate_song_before > 0:
        df["_score"] = df.apply(completeness_score,axis=1)
        df = (
            df
            .sort_values("_score", ascending=False)
            .drop_duplicates(subset=["name", "artists"])
            .drop(columns="_score")
        )

    repaired_rows = len(df)

    output_file = OUTPUT_DIR / f"{dataset_name.lower()}_repaired.csv"

    df.to_csv(output_file, index=False)

    # Report
    write(f"Rows Loaded                    : {original_rows:,}")
    write(f"Removed (Missing ID/Name)      : {removed_missing_id_name:,}")

    if dataset_name == "900K":
        write(f"Removed (Missing Artists)      : {removed_missing_artists:,}")

    write(f"Duplicate IDs Found            : {duplicate_ids_before:,}")
    write(f"Duplicate Song+Artist Found    : {duplicate_song_before:,}")
    write(f"Final Rows                     : {repaired_rows:,}")
    write(f"Saved To                       : {output_file}")
    write()


# Main
def main():

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    for dataset_name, path in FILES.items():
        repair_dataset(dataset_name, path)

    print("\nRepair complete.")
    print(f"Report saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()