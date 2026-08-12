import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from app.data_loader import SpotifyDataLoader
from app.preprocessing import SpotifyPreprocessor
from app.services.search_service import SearchService
from app.services.matching_service import MatchingService


# ---------------------------------------
# Load dataset (same as main.py)
# ---------------------------------------

loader = SpotifyDataLoader().load_data()

processor = (
    SpotifyPreprocessor(loader.song_df)
    .clean_data()
    .prepare_audio_features()
)

songs = processor.get_dataframe()

print(f"\nDataset Loaded : {len(songs):,} songs")


# ---------------------------------------
# Initialize services
# ---------------------------------------

search = SearchService(songs)
matcher = MatchingService()


# ---------------------------------------
# Spotify track (simulated)
# ---------------------------------------

spotify_track = {
    "name": "Believer",
    "artist": "Imagine Dragons",
    "release_date": "2017-02-01"
}
# spotify_track = {
#     "name": "Hotel California",
#     "artist": "Eagles",
#     "release_date": "1976-12-08"
# }



# ---------------------------------------
# Stage 1
# ---------------------------------------

print("\n========== Stage 1 ==========")

candidates = search.get_candidates(
    spotify_track["name"],
    spotify_track["artist"]
)

print(f"Candidates Found : {len(candidates)}")

print()

print(
    candidates[
        [
            "name",
            "artists",
            "year"
        ]
    ].head(10)
)


# ---------------------------------------
# Stage 2
# ---------------------------------------

print("\n========== Stage 2 ==========")

matches = matcher.find_best_match(
    spotify_track,
    candidates
)

print()

print(
    matches[
        [
            "name",
            "artists",
            "year",
            "matching_score"
        ]
    ]
)