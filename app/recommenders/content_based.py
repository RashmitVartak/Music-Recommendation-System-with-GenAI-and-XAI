import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.services.feature_service import FeatureService


class ContentBasedRecommender:

    def __init__(self, df):

        self.df = df.reset_index(drop=True)
        self.feature_service = FeatureService()
        self.feature_matrix = (
            self.feature_service.create_feature_matrix(self.df)
        )

    def available_songs(self):
        return sorted(self.df["name"].unique())

    def get_song_index(self, song_name):

        matches = self.df[
            self.df["name"].str.lower() == song_name.lower()
        ]

        if matches.empty:
            return None

        return matches.index[0]

    def recommend(self, song_name, n=10):

        index = self.get_song_index(song_name)

        if index is None:
            return None

        query_vector = self.feature_matrix[index].reshape(1, -1)

        similarity_scores = cosine_similarity(
            query_vector,
            self.feature_matrix
        ).flatten()

        sorted_indices = similarity_scores.argsort()[::-1]

        sorted_indices = sorted_indices[1:n+1]

        recommendations = (self.df.iloc[sorted_indices].copy())

        recommendations["Similarity Score"] = (similarity_scores[sorted_indices])

        recommendations["Similarity Score"] = (recommendations["Similarity Score"].round(3))

        return recommendations[
            [
                "name",
                "artists",
                "year",
                "popularity",
                "Similarity Score",
            ]
        ]