import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.feature_service import FeatureService

class ContentBasedRecommender:
    MIN_SIMILARITY = 0.01

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.df["_name_lower"] = (self.df["name"]
                                         .fillna("")
                                         .astype(str)
                                         .str.lower()
                                        )
        self.feature_service = FeatureService()
        self.feature_matrix = (self.feature_service.create_feature_matrix(self.df))

        norms=np.linalg.norm(self.feature_matrix,
                            axis=1, 
                            keepdims=True)

        self.normalized_feature_matrix=(
            self.feature_matrix/np.maximum(norms,1e-12)
        )

    def available_songs(self):
        """Checks for available songs in the dataset."""
        return sorted(self.df["name"].dropna().unique())

    def get_song_index(self, song_name):
        matches = self.df[self.df["_name_lower"] == song_name.lower()]

        if matches.empty:
            return None

        return matches.index[0]

    def recommend(self,song_name,n=10):

        index = self.get_song_index(song_name)
        if index is None:
            return None
        
        query_vector=self.normalized_feature_matrix[index]
        similarity_scores=self.normalized_feature_matrix @ query_vector

        sorted_indices = similarity_scores.argsort()[::-1]

        # Remove the query song
        sorted_indices = [
            i for i in sorted_indices 
            if i != index
        ]

        # Keep only sufficiently similar songs
        sorted_indices = [
            i for i in sorted_indices
            if similarity_scores[i] >= self.MIN_SIMILARITY
        ]

        # Return only top n
        sorted_indices = sorted_indices[:n]

        recommendations = (self.df.iloc[sorted_indices].copy())

        # recommendations["score"] = (self.normalize_scores(
        #                                 recommendations["score"]
        #                                 ).round(3)
        #                         )

        # no more normalized score for selected recommendation
        recommendations["score"] = (similarity_scores[sorted_indices].round(3))

        recommendations["source"] = "Content"

        return recommendations[
            [ "id","name","artists","year","popularity","score","source"]
        ]