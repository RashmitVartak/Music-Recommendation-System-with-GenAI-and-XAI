from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets"

SONG_DATA = DATASET_DIR / "data.csv"
ARTIST_DATA = DATASET_DIR / "data_by_artist.csv"
GENRE_DATA = DATASET_DIR / "data_by_genres.csv"
YEAR_DATA = DATASET_DIR / "data_by_year.csv"
GENRE_SONG_DATA = DATASET_DIR / "data_w_genres.csv"