import pandas as pd

class HybridRecommender:

    def __init__(self,content_recommender,collaborative_recommender,content_weight=0.6,collaborative_weight=0.4,):

        self.content = content_recommender
        self.collaborative = collaborative_recommender
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

    def set_weights(self,content_weight,collaborative_weight):

            self.content_weight = content_weight
            self.collaborative_weight = collaborative_weight

    def create_merge_key(self, df):

        df = df.copy()

        df["merge_key"] = (

            df["name"]
            .fillna("")
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.strip()

            +

            "|"

            +

            df["artists"]
            .fillna("")
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.strip()
    )

        return df
    
    def recommend(self,song_name,top_n=10):
        content = self.content.recommend(song_name,top_n * 2)
        collaborative = self.collaborative.recommend(song_name,top_n * 2)

        if content.empty and collaborative.empty:
            return None

        if content.empty:
            collaborative["source"] = "Hybrid"
            return collaborative.head(top_n)

        if collaborative.empty:
            content["source"] = "Hybrid"
            return content.head(top_n)


        content = self.create_merge_key(content)
        collaborative = self.create_merge_key(collaborative)

        # Rename score columns before merge
        content = content.rename(
            columns={
                "score": "content_score",
                "id": "id_content"
            }
        )

        collaborative = collaborative.rename(
            columns={
                "score": "collaborative_score",
                "id": "id_collaborative"
            }
        )

        # Merge recommendations
        hybrid = pd.merge(
            content,
            collaborative,
            on="merge_key",
            how="outer",
            suffixes=("_content", "_collaborative")
        )

        hybrid["id"] = hybrid["id_content"].combine_first(hybrid["id_collaborative"])

        hybrid["name"] = hybrid["name_content"].combine_first(hybrid["name_collaborative"])

        hybrid["artists"] = hybrid["artists_content"].combine_first(hybrid["artists_collaborative"])

        hybrid["year"] = hybrid["year_content"].combine_first(hybrid["year_collaborative"])

        hybrid["popularity"] = hybrid["popularity_content"].combine_first(hybrid["popularity_collaborative"])

        hybrid["score"] = (
            self.content_weight*hybrid["content_score"].fillna(0)
            +
            self.collaborative_weight*hybrid["collaborative_score"].fillna(0)
        )

        hybrid = hybrid.sort_values("score",ascending=False)

        return hybrid[
                [
                    "id",
                    "name",
                    "artists",
                    "content_score",
                    "collaborative_score",
                    "score",
                    "year",
                    "popularity",
                    "source"
                ]
            ].head(top_n)      