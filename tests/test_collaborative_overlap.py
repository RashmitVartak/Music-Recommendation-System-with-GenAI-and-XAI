import os
import sys
import re
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# Paths

CONTENT_PATH = "datasets/processed/merged_dataset.csv"
SONG_DATA_PATH = "datasets/song_data.csv"
TRIPLETS_PATH = "datasets/triplets_file.csv"


# Load datasets

content = pd.read_csv(CONTENT_PATH)

song_data = pd.read_csv(SONG_DATA_PATH,
    usecols=["song_id", "title", "artist_name"])

triplets = pd.read_csv(TRIPLETS_PATH,usecols=["song_id"])

print("=" * 70)
print("DATASET SIZES")
print("=" * 70)

print(f"Content dataset       : {len(content):,}")
print(f"Song metadata         : {len(song_data):,}")
print(f"Triplet interactions  : {len(triplets):,}")

# Get songs that actually have user interactions
active_song_ids = set(triplets["song_id"].dropna().unique())
collab_active = song_data[song_data["song_id"].isin(active_song_ids)].copy()

print("\n" + "=" * 70)
print("COLLABORATIVE ACTIVE SONGS")
print("=" * 70)

print(f"Unique songs in triplets       : "f"{len(active_song_ids):,}")

print(f"Active songs in song_data      : "f"{len(collab_active):,}")

# Normalization
def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = value.replace("&", "and")

    # Remove remaster / remastered suffixes
    value = re.sub(
        r"\s*[-–—]\s*(?:\d{4}\s+)?remaster(?:ed)?\b.*$",
        "",
        value
    )

    # Remove live/remaster information in brackets
    value = re.sub(
        r"\s*\([^)]*(?:remaster|remastered|live)[^)]*\)",
        "",
        value
    )

    # Normalize punctuation
    value = re.sub(r"[^\w\s]", " ", value)
    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()

# Normalize Content

content["title_norm"] = content["name"].apply(normalize_text)
content["artist_norm"] = content["artists"].apply(normalize_text)

# Normalize Collaborative active songs
collab_active["title_norm"] = collab_active["title"].apply(normalize_text)
collab_active["artist_norm"] = collab_active["artist_name"].apply(normalize_text)


# Create matching keys
content["match_key"] = (
    content["title_norm"]
    + " || "
    + content["artist_norm"]
)

collab_active["match_key"] = (
    collab_active["title_norm"]
    + " || "
    + collab_active["artist_norm"]
)


# Unique keys
content_keys = set(
    content.loc[
        content["match_key"] != " || ",
        "match_key"
    ]
)

collab_active_keys = set(
    collab_active.loc[
        collab_active["match_key"] != " || ",
        "match_key"
    ]
)


# Actual overlap

overlap = content_keys.intersection(collab_active_keys)

content_only = content_keys - collab_active_keys
collab_only = collab_active_keys - content_keys


# Results

print("\n" + "=" * 70)
print("CONTENT ↔ ACTIVE COLLABORATIVE OVERLAP")
print("=" * 70)

print(f"Unique Content songs              : "f"{len(content_keys):,}")
print(f"Active Collaborative songs        : "f"{len(collab_active_keys):,}")
print(f"Songs present in BOTH              : "f"{len(overlap):,}")

# Coverage percentages

content_coverage = (
    len(overlap) / len(content_keys) * 100
    if content_keys else 0
)

collab_coverage = (
    len(overlap) / len(collab_active_keys) * 100
    if collab_active_keys else 0
)


print(
    f"\nContent songs with collaborative "
    f"interactions : {content_coverage:.2f}%"
)

print(
    f"Active collaborative songs also in "
    f"Content       : {collab_coverage:.2f}%"
)


# Comparison with previous metadata overlap

print("\n" + "=" * 70)
print("COVERAGE INTERPRETATION")
print("=" * 70)

print("This measures the songs that can actually participate ""in Collaborative recommendations.")

print(f"\nContent-only songs                    : "f"{len(content_only):,}")

print(f"Collaborative-active-only songs      : "f"{len(collab_only):,}")


# Show sample overlap
print("\n" + "=" * 70)
print("SAMPLE ACTIVE COLLABORATIVE OVERLAP")
print("=" * 70)

for key in list(overlap)[:20]:
    print(key)


# Save overlap for inspection
overlap_df = content[
    content["match_key"].isin(overlap)
][
    [
        "id",
        "name",
        "artists",
        "year"
    ]
].copy()

overlap_df.to_csv(
    "tests/content_active_collaborative_overlap.csv",
    index=False
)

print(
    "\nOverlap file created:"
    "\ntests/content_active_collaborative_overlap.csv"
)