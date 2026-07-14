import pandas as pd

class PopularityRecommender:
    def __init__(self, df):
        self.df = df.copy()

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


    def recommend(self, top_n=10):
        recommendations = (self.df.sort_values(
                by="popularity",ascending=False
            ).head(top_n).copy()
        )

        recommendations["score"] = (
            self.normalize_scores(
                recommendations["popularity"]).round(3)
                )

        recommendations["source"] = ("Popularity")
        
        print("Columns:", recommendations.columns.tolist())
        print(recommendations.head())

        return recommendations[
            [ "id","name","artists","year","popularity","score","source"]
        ]