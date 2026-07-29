import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.services.feature_service import FeatureService

class ContentBasedRecommender:
    MIN_SIMILARITY = 0.01

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.feature_service = FeatureService()
        self.feature_matrix = (self.feature_service.create_feature_matrix(self.df))

    def available_songs(self):
        """Checks for available songs in the dataset."""
        return sorted(self.df["name"].dropna().unique())

    def get_song_index(self,song_name):
        """Returns the index of the song in the dataset by matching the song name."""
        matches = self.df[self.df["name"].str.lower() == song_name.lower()]

        if matches.empty:
            return None

        return matches.index[0]

    def normalize_scores(self,scores):
        scores = pd.Series(scores)

        if scores.max() == scores.min():
            return pd.Series(
                [1.0] * len(scores),
                index=scores.index
            )

        return (
            (scores - scores.min())
            /
            (scores.max() - scores.min())
        )

    def recommend(self,song_name,n=10):

        index = self.get_song_index(song_name)
        if index is None:
            return None
        
        query_vector = (self.feature_matrix[index].reshape(1, -1))
        similarity_scores = cosine_similarity(query_vector,self.feature_matrix).flatten()
        sorted_indices = similarity_scores.argsort()[::-1]

        # Remove the query song
        sorted_indices = [i for i in sorted_indices if i != index]

        # Keep only sufficiently similar songs
        sorted_indices = [
            i for i in sorted_indices
            if similarity_scores[i] >= self.MIN_SIMILARITY
        ]

        # Return only top n
        sorted_indices = sorted_indices[:n]

        recommendations = (self.df.iloc[sorted_indices].copy())

        recommendations["score"] = (similarity_scores[sorted_indices])

        # recommendations["score"] = (self.normalize_scores(
        #                                 recommendations["score"]
        #                                 ).round(3)
        #                         )
        # no more normalized score for selected recommendation
        recommendations["score"] = (similarity_scores[sorted_indices].round(3))

        recommendations["source"] = ("Content")

        return recommendations[
            [ "id","name","artists","year","popularity","score","source"]
        ]