"""
Centralized column mappings for all supported datasets.

Every dataset is first renamed to the canonical schema
before any preprocessing or merging.
"""
# 114K DATASET
DATASET_114K_MAPPING = {
    "track_id": "id",
    "track_name": "name",
}

# 900K DATASET
DATASET_900K_MAPPING = {
    "track_id": "id",
    "track_artists": "artists",
    "album_release_date": "release_date",
}

# Columns we ultimately want in the merged dataset
CANONICAL_COLUMNS = [
    "id",
    "name",
    "artists",
    "album_name",
    "release_date",
    "year",
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "genres",
    "artist_popularity",
    "artist_followers",
]