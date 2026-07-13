import pandas as pd

from config import *

class SpotifyDataLoader:

    def __init__(self):
        self.song_df = None
        self.artist_df = None
        self.genre_df = None
        self.year_df = None
        self.genre_song_df = None

    def load_data(self):

        self.song_df = pd.read_csv(SONG_DATA)
        self.artist_df = pd.read_csv(ARTIST_DATA)
        self.genre_df = pd.read_csv(GENRE_DATA)
        self.year_df = pd.read_csv(YEAR_DATA)
        self.genre_song_df = pd.read_csv(GENRE_SONG_DATA)

        return self