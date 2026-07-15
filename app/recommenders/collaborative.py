import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:

    def __init__(self,triplets_path, song_data_path):

        # Load datasets
        self.triplets = pd.read_csv(triplets_path)
        self.song_data = pd.read_csv(song_data_path)

        # Merge
        self.dataset = pd.merge(self.triplets,self.song_data,on="song_id",how="left")

        # Remove duplicate song metadata
        self.dataset = self.dataset.drop_duplicates(subset=["song_id"])

        # Build song-user matrix
        self.song_user_matrix = (self.triplets.pivot_table(
                                                    index="song_id",
                                                    columns="user_id",
                                                    values="listen_count",
                                                    fill_value=0
                                                )
                                              )

        # Sparse matrix
        self.sparse_matrix = csr_matrix(self.song_user_matrix.values)

    def available_songs(self):
        return sorted(self.dataset["title"].dropna().unique())

    def get_song_id(self, song_name):
        result = self.dataset[self.dataset["title"].str.lower()== song_name.lower()]

        if result.empty:
            return None

        return result.iloc[0]["song_id"]
    
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

        song_id = self.get_song_id(song_name)
        if (song_id is None or song_id not in self.song_user_matrix.index):

            return pd.DataFrame(
                columns=[ "id","name","artists","year","popularity","score","source"]
                )
           
        idx = self.song_user_matrix.index.get_loc(song_id)

        # Only compare ONE song against all songs
        similarity_scores = cosine_similarity(self.sparse_matrix[idx],self.sparse_matrix).flatten()
        similar_indices = (similarity_scores.argsort()[::-1])[1:n+1]
        similar_song_ids = (self.song_user_matrix.index[similar_indices])

        recommendations = (
            self.dataset[
                self.dataset["song_id"]
                .isin(similar_song_ids)]
                .drop_duplicates("song_id").copy()
        )

        recommendations["score"] = (similarity_scores[similar_indices])
        recommendations["score"] = (self.normalize_scores(recommendations["score"]).round(3))

         # This dataset has no popularity column
        recommendations["popularity"] = None

        recommendations["source"] = ("Collaborative")

        

        return recommendations[
            [ "song_id","title","artist_name","year","popularity","score","source"]
        ].rename(
            columns={
                "song_id": "id",
                "title": "name",
                "artist_name": "artists"
            }
        )