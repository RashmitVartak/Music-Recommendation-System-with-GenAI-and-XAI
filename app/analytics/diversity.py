import pandas as pd


class RecommendationDiversity:

    @staticmethod
    def artist_diversity(df):
        if df.empty:
            return 0

        return round((df["artists"].nunique()/len(df)) * 100, 2)

    @staticmethod
    def year_diversity(df):
        if df.empty:
            return 0

        years = pd.to_numeric(df["year"],errors="coerce").dropna()

        if len(years) <= 1:
            return 0

        spread = years.max() - years.min()

        return min(round((spread / 20) * 100, 2),100)

    @staticmethod
    def popularity_diversity(df):
        if df.empty:
            return 0

        popularity = pd.to_numeric(df["popularity"],errors="coerce").dropna()

        if len(popularity) <= 1:
            return 0

        spread = popularity.max() - popularity.min()

        return min(round(spread, 2),100)

    @staticmethod
    def diversity_summary(df):

        artist = RecommendationDiversity.artist_diversity(df)
        year = RecommendationDiversity.year_diversity(df)
        popularity = RecommendationDiversity.popularity_diversity(df)

        overall = round((artist+year+popularity) / 3, 2)

        return {
            "artist_diversity": artist,
            "year_diversity": year,
            "popularity_diversity": popularity,
            "overall_diversity": overall
        }