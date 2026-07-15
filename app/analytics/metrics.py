import pandas as pd


class RecommendationMetrics:

    @staticmethod
    def total_recommendations(df):
        return len(df)

    @staticmethod
    def unique_artists(df):
        if df.empty:
            return 0

        return df["artists"].nunique()

    @staticmethod
    def average_score(df):
        if df.empty:
            return 0

        return round(df["score"].mean() * 100, 2)

    @staticmethod
    def average_popularity(df):
        if df.empty:
            return 0

        popularity = df["popularity"].dropna()
        if popularity.empty:
            return None

        return round(popularity.mean(), 2)

    @staticmethod
    def average_year(df):
        if df.empty:
            return 0

        year = pd.to_numeric(df["year"],errors="coerce").dropna()
        if year.empty:
            return None

        return int(year.mean())

    @staticmethod
    def unique_years(df):
        if df.empty:
            return 0

        return df["year"].nunique()

    @staticmethod
    def recommendation_summary(df):
        return {

            "total_songs":RecommendationMetrics.total_recommendations(df),

            "unique_artists":RecommendationMetrics.unique_artists(df),

            "average_score":RecommendationMetrics.average_score(df),

            "average_popularity":RecommendationMetrics.average_popularity(df),

            "average_year":RecommendationMetrics.average_year(df),

            "unique_years":RecommendationMetrics.unique_years(df)
        }