import pandas as pd

class PopularityRecommender:
    def __init__(self, df):
        self.df = df.copy()

    def recommend(self, top_n=10):
        recommendations = (self.df.sort_values(
                by="popularity",ascending=False
            ).head(top_n)
        )

        return recommendations[
            [
                "name",
                "artists",
                "year",
                "popularity"
            ]
        ]