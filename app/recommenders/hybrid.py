import pandas as pd

class HybridRecommender:

    def __init__(self,content_recommender,collaborative_recommender,content_weight=0.6,collaborative_weight=0.4,):

        self.content = content_recommender
        self.collaborative = collaborative_recommender
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

    def recommend(self,song_name,top_n=10):
        content_df = self.content.recommend(song_name,top_n * 2)
        collaborative_df = self.collaborative.recommend(song_name,top_n * 2)

        if content_df is None and collaborative_df is None:
            return None

        if content_df is None:
            return collaborative_df.head(top_n)

        if collaborative_df is None:
            return content_df.head(top_n)

        # Rename similarity columns
        content_df = content_df.rename(
            columns={"Similarity Score": "Content Score"}
        )

        collaborative_df = collaborative_df.rename(
            columns={
                "Similarity": "Collaborative Score",
                "title": "name",
                "artist_name": "artists"
            }
        )

        # Merge recommendations
        hybrid = pd.merge(
            content_df,
            collaborative_df[
                [
                    "name",
                    "Collaborative Score"
                ]
            ],
            on="name",
            how="outer")

        hybrid["Content Score"] = hybrid["Content Score"].fillna(0)

        hybrid["Collaborative Score"] = hybrid["Collaborative Score"].fillna(0)

        # Weighted average
        hybrid["Hybrid Score"] = (
            self.content_weight* hybrid["Content Score"]
            +
            self.collaborative_weight* hybrid["Collaborative Score"]
)

        hybrid = hybrid.sort_values("Hybrid Score",ascending=False)

        return hybrid.head(top_n)