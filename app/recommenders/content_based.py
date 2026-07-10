import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.utils import AUDIO_FEATURES

class ContentBasedRecommender:

    def __init__(self, df):

        self.df = df.copy()
        self.similarity_matrix = None
    
    def build_similarity_matrix(self):

        self.similarity_matrix = cosine_similarity(
            self.df[AUDIO_FEATURES]
        )

        return self

    def get_song_index(self, song_name):

        song = self.df[
            self.df["name"].str.lower()
            == song_name.lower()
        ]

        if song.empty:
            return None

        return song.index[0]
    
    def recommend(self, song_name, n=10):

        index = self.get_song_index(song_name)

        if index is None:
            return None

        similarities = list(
            enumerate(
                self.similarity_matrix[index]
            )
        )

        similarities = sorted(
            similarities,
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = similarities[1:n+1]

        return self.df.iloc[
            [i[0] for i in recommendations]
        ][
            [
                "name",
                "artists",
                "year",
                "popularity"
            ]
        ]