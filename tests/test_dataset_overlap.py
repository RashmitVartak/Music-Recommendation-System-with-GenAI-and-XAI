import os
import sys
import re
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# ============================================================
# Paths
# ============================================================

CONTENT_PATH = "datasets/processed/merged_dataset.csv"
COLLAB_PATH = "datasets/song_data.csv"


# ============================================================
# Load datasets
# ============================================================

content = pd.read_csv(CONTENT_PATH)
collab = pd.read_csv(COLLAB_PATH)


print("=" * 70)
print("DATASET SIZES")
print("=" * 70)

print(f"Content dataset       : {len(content):,}")
print(f"Collaborative dataset : {len(collab):,}")


# ============================================================
# Normalization
# ============================================================

def normalize_text(value):
    """
    Normalize text so that small formatting differences
    don't prevent a match.
    """

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Normalize '&' and 'and'
    value = value.replace("&", "and")

    # Remove remaster / remastered information
    value = re.sub(
        r"\s*[-–—]\s*(?:\d{4}\s+)?remaster(?:ed)?\b.*$",
        "",
        value
    )

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


# ============================================================
# Create normalized keys
# ============================================================

content["title_norm"] = content["name"].apply(
    normalize_text
)

content["artist_norm"] = content["artists"].apply(
    normalize_text
)

collab["title_norm"] = collab["title"].apply(
    normalize_text
)

collab["artist_norm"] = collab["artist_name"].apply(
    normalize_text
)


# ============================================================
# Create title + artist key
# ============================================================

content["match_key"] = (
    content["title_norm"]
    + " || "
    + content["artist_norm"]
)

collab["match_key"] = (
    collab["title_norm"]
    + " || "
    + collab["artist_norm"]
)


# ============================================================
# Unique keys
# ============================================================

content_keys = set(
    content.loc[
        content["match_key"] != " || ",
        "match_key"
    ]
)

collab_keys = set(
    collab.loc[
        collab["match_key"] != " || ",
        "match_key"
    ]
)


# ============================================================
# Exact normalized overlap
# ============================================================

overlap = content_keys.intersection(collab_keys)


print("\n" + "=" * 70)
print("CONTENT ↔ COLLABORATIVE OVERLAP")
print("=" * 70)

print(f"Unique Content songs          : {len(content_keys):,}")
print(f"Unique Collaborative songs    : {len(collab_keys):,}")
print(f"Songs in both datasets        : {len(overlap):,}")


content_coverage = (
    len(overlap) / len(content_keys) * 100
    if content_keys else 0
)

collab_coverage = (
    len(overlap) / len(collab_keys) * 100
    if collab_keys else 0
)

print(
    f"\nContent covered by Collaborative : "
    f"{content_coverage:.2f}%"
)

print(
    f"Collaborative covered by Content : "
    f"{collab_coverage:.2f}%"
)


# ============================================================
# Show examples of overlap
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE OVERLAPPING SONGS")
print("=" * 70)

sample_keys = list(overlap)[:20]

for key in sample_keys:
    print(key)


# ============================================================
# Songs only in Content
# ============================================================

content_only = content_keys - collab_keys

print("\n" + "=" * 70)
print("CONTENT-ONLY")
print("=" * 70)

print(
    f"Songs only in Content : "
    f"{len(content_only):,}"
)


# ============================================================
# Songs only in Collaborative
# ============================================================

collab_only = collab_keys - content_keys

print("\n" + "=" * 70)
print("COLLABORATIVE-ONLY")
print("=" * 70)

print(
    f"Songs only in Collaborative : "
    f"{len(collab_only):,}"
)


# ============================================================
# Save overlapping songs for inspection
# ============================================================

overlap_df = content[
    content["match_key"].isin(overlap)
][
    [
        "id",
        "name",
        "artists",
        "year",
        "match_key"
    ]
].copy()

overlap_df.to_csv(
    "tests/content_collaborative_overlap.csv",
    index=False
)

print("\nOverlap file created:")
print("tests/content_collaborative_overlap.csv")