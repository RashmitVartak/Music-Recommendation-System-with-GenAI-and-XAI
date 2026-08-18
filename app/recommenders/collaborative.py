import pandas as pd
import re

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:

    @staticmethod
    def normalize_title(value):
        value = str(value).lower().strip()

        value = re.sub(
            r"\s*[-–—]\s*(?:\d{4}\s+)?remaster(?:ed)?\b.*$",
            "",
            value
        )

        value = re.sub(
            r"\s*[-–—]\s*live\b.*$",
            "",
            value
        )

        value = re.sub(
            r"\s*\([^)]*(?:remaster|remastered|live)[^)]*\)",
            "",
            value
        )

        value = re.sub(r"[^\w\s]", " ", value)
        value = re.sub(r"\s+", " ", value)

        return value.strip()


    def __init__(self,triplets_path, song_data_path):

        # Load datasets
        self.triplets = pd.read_csv(triplets_path)
        self.song_data = pd.read_csv(song_data_path)

        # Merge
        self.dataset = pd.merge(self.triplets,self.song_data,on="song_id",how="left")

        # Remove duplicate song metadata
        self.dataset = self.dataset.drop_duplicates(subset=["song_id"])

        self.dataset["normalized_title"] = (
                self.dataset["title"]
                .fillna("")
                .astype(str)
                .apply(self.normalize_title)
            )

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

    def has_interactions(self, song_id):
        return (
            song_id is not None
            and song_id in self.song_user_matrix.index
        )

    def resolve_song_id(self, song_name, artist_name=""):
        """
        Resolve a Spotify/Content song to the corresponding
        song_id in the Collaborative dataset.
        """

        if not song_name:
            return None

        title = str(song_name).lower().strip()
        artist = str(artist_name).lower().strip()

        # 1. Exact title + artist
        title_series = (
            self.dataset["title"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        artist_series = (
            self.dataset["artist_name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        exact_mask = title_series.eq(title)

        if artist:
            exact_mask &= artist_series.eq(artist)

        result = self.dataset[exact_mask]

        if not result.empty:
            return result.iloc[0]["song_id"]

        
        #2. Normalized input
        normalized_input = self.normalize_title(title)

        dataset_normalized_titles = (
            self.dataset["title"]
            .fillna("")
            .astype(str)
            .apply(self.normalize_title)
        )

        # 3. Normalized title + artist
        normalized_mask = (self.dataset["normalized_title"]==normalized_input)

        if artist:
            normalized_mask &= artist_series.eq(artist)

        result = self.dataset[normalized_mask]

        if not result.empty:
            return result.iloc[0]["song_id"]

        # 4. Normalized title only
        result = self.dataset[self.dataset["normalized_title"]==normalized_input]
        if not result.empty:
            return result.iloc[0]["song_id"]

        return None

    def recommend_by_id(self, song_id, n=10):
        """
        Generate collaborative recommendations using
        an already-resolved Collaborative song_id.
        """

        if (song_id is None or song_id not in self.song_user_matrix.index):
            return pd.DataFrame(
                columns=[ "id","name","artists","year","popularity","score","source"]
               )

        # Get the row position of the selected song
        idx = self.song_user_matrix.index.get_loc(song_id)

        # Calculate similarity against all songs
        similarity_scores = cosine_similarity(self.sparse_matrix[idx],self.sparse_matrix
                                            ).flatten()

        # Get top N similar songs
        similar_indices = (similarity_scores.argsort()[::-1]
                            )[1:n + 1]

        similar_song_ids = (self.song_user_matrix.index[similar_indices])

        # Get metadata for those songs
        recommendations = (
            self.dataset[self.dataset["song_id"].isin(similar_song_ids)]
            .drop_duplicates("song_id")
            .copy()
        )

        recommendations["score"] = (similarity_scores[similar_indices])

        recommendations["score"] = (
            self.normalize_scores(recommendations["score"]).round(3)
        )

        recommendations["popularity"] = None
        recommendations["source"] = "Collaborative"

        return recommendations[
                [ "song_id","title","artist_name","year","popularity","score","source"]
            ].rename(
            columns={"song_id": "id",
                    "title": "name",
                    "artist_name": "artists",}
        )

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

    def recommend(self, song_name, n=10):

        song_id = self.get_song_id(song_name)

        return self.recommend_by_id(song_id, n)