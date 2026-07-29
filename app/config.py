from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

DATASET_DIR = BASE_DIR / "datasets"

SONG_DATA = DATASET_DIR / "data.csv"
# SONG_DATA = DATASET_DIR / "processed" / "merged_dataset.csv"
ARTIST_DATA = DATASET_DIR / "data_by_artist.csv"

GENRE_DATA = DATASET_DIR / "data_by_genres.csv"
YEAR_DATA = DATASET_DIR / "data_by_year.csv"
GENRE_SONG_DATA = DATASET_DIR / "data_w_genres.csv"