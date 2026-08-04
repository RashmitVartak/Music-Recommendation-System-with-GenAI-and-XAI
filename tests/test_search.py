import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from app.data_loader import SpotifyDataLoader
from app.preprocessing import SpotifyPreprocessor
from app.services.search_service import SearchService

loader = SpotifyDataLoader().load_data()

processor = (
    SpotifyPreprocessor(loader.song_df)
    .clean_data()
    .prepare_audio_features()
)

songs = processor.get_dataframe()

search = SearchService(songs)

query = input("Search Song: ")

results = search.search(query)

print("\nResults Found:", len(results))

print(
    results[
        [
            "name",
            "artists",
            "year",
            "popularity"
        ]
    ]
)

print("\n" + "=" * 60)
print("Candidate Generation")
print("=" * 60)

candidates = search.get_candidates(
    title=query,
    artist="Imagine Dragons"
)

print("Candidates:", len(candidates))

print(
    candidates[
        [
            "name",
            "artists",
            "year"
        ]
    ].head(20)
)