from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets"

FILES = {
    "Current": DATASET_DIR / "data.csv",
    "114K": DATASET_DIR / "processed" / "114k_repaired.csv",
    "900K": DATASET_DIR / "processed" / "900k_repaired.csv",
}

PRIORITY = {
    "Current": 3,
    "114K": 2,
    "900K": 1,
}

OUTPUT_DIR = DATASET_DIR / "processed"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "merge_report.txt"


# -------------------------------------------------------
# Report helper
# -------------------------------------------------------

def write(text=""):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(str(text))
        f.write("\n")


# -------------------------------------------------------
# Standardize artists
# -------------------------------------------------------

def normalize_artists(value):

    if pd.isna(value):
        return pd.NA

    value = str(value)

    value = (
        value.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    artists = [
        artist.strip()
        for artist in value.replace(",", ";").split(";")
        if artist.strip()
    ]

    return "; ".join(sorted(set(artists)))


# -------------------------------------------------------
# Standardize song names
# -------------------------------------------------------

def normalize_name(value):

    if pd.isna(value):
        return pd.NA

    return " ".join(str(value).strip().split()).lower()


# -------------------------------------------------------
# Completeness score
# -------------------------------------------------------

IMPORTANT_COLUMNS = [
    "album_name",
    "release_date",
    "year",
    "genres",
    "artist_popularity",
    "artist_followers",
]


def completeness_score(row):

    score = 0

    for col in IMPORTANT_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            score += 1

    return score


# Main

# -------------------------------------------------------
# Enrich primary record using lower-priority duplicate
# -------------------------------------------------------
def enrich_record(primary, secondary):
    for col in primary.index:
        if col in {"id","name","artists","source_dataset","dataset_priority","normalized_name","completeness_score"}:
            continue
        if ((str(primary[col])=="" ) or (primary[col]!=primary[col])) and not (secondary[col]!=secondary[col]) and str(secondary[col])!="":
            primary[col]=secondary[col]
    return primary

def main():

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    frames = []

    write("=" * 80)
    write("MERGE REPORT")
    write("=" * 80)
    write()

    for dataset_name, file in FILES.items():

        print(f"Loading {dataset_name}...")

        df = pd.read_csv(file, low_memory=False)


        # Standardize release date
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"],errors="coerce")

        df["source_dataset"] = dataset_name
        df["dataset_priority"] = PRIORITY[dataset_name]

        df["artists"] = df["artists"].apply(normalize_artists)
        df["normalized_name"] = df["name"].apply(normalize_name)

        frames.append(df)

        write(f"{dataset_name:<12}: {len(df):,} rows")

    merged = pd.concat(frames, ignore_index=True)
    
    # Fill missing popularity
    if "popularity" in merged.columns:
        merged["popularity"] = merged["popularity"].fillna(-1)

    # Recover missing year
    if "year" in merged.columns and "release_date" in merged.columns:
        mask = (merged["year"].isna() & merged["release_date"].notna())
        merged.loc[mask, "year"] = (merged.loc[mask, "release_date"].dt.year)

    rows_before_merge = len(merged)

    write()
    write(f"Rows Before Merge          : {rows_before_merge:,}")

    merged["completeness_score"] = merged.apply(completeness_score,axis=1)

    # Keep best duplicate by ID
    merged = merged.sort_values(
        by=["dataset_priority","completeness_score","popularity","year",],
        ascending=False,
    )

    duplicate_ids = merged["id"].duplicated().sum()
    enriched=[]
    for _,grp in merged.groupby("id",sort=False):
        p=grp.iloc[0].copy()
        for _,r in grp.iloc[1:].iterrows():
            p=enrich_record(p,r)
        enriched.append(p)
    merged=pd.DataFrame(enriched)
    rows_after_id=len(merged)

    # Keep best duplicate by Song + Artist
    duplicate_song_artist = merged.duplicated(subset=["normalized_name", "artists"]).sum()

    song_rows=[]
    duplicate_song_artist=0
    for _,grp in merged.groupby(["normalized_name","artists"],dropna=False,sort=False):
        p=grp.iloc[0].copy()
        duplicate_song_artist+=max(len(grp)-1,0)
        for _,r in grp.iloc[1:].iterrows():
            p=enrich_record(p,r)
        song_rows.append(p)
    merged=pd.DataFrame(song_rows)
    rows_after_song=len(merged)

    merged = merged.drop(
        columns=[
            "normalized_name",
            "dataset_priority",
            "completeness_score",
        ]
    )

    output_file = OUTPUT_DIR / "merged_dataset.csv"

    merged.to_csv(output_file, index=False)

    write()
    write("=" * 80)
    write("MERGE SUMMARY")
    write("=" * 80)

    write(f"Current Dataset Rows           : {len(frames[0]):,}")
    write(f"114K Dataset Rows              : {len(frames[1]):,}")
    write(f"900K Dataset Rows              : {len(frames[2]):,}")

    write("-" * 80)

    write(f"Rows Before Merge             : {rows_before_merge:,}")
    write(f"Rows After ID Dedup           : {rows_after_id:,}")
    write(f"Rows After Song Dedup         : {rows_after_song:,}")

    write("-" * 80)

    write(f"Duplicate IDs Removed         : {duplicate_ids:,}")
    write(f"Duplicate Song+Artist Removed : {duplicate_song_artist:,}")

    write("-" * 80)

    write(f"Final Dataset Size            : {len(merged):,}")
    write()
    write(f"Saved To:")
    write(output_file)

    print("\nMerge complete.")
    print(f"Merged dataset saved to:\n{output_file}")
    print(f"Report saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()