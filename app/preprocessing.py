import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.utils import AUDIO_FEATURES


class SpotifyPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df = self.df.dropna(subset=AUDIO_FEATURES)
        self.df["year"] = (pd.to_numeric(self.df["year"], errors="coerce"))
        self.df["popularity"] = (pd.to_numeric(self.df["popularity"], errors="coerce"))

        return self

    def prepare_audio_features(self):
        scaler = StandardScaler()
        self.df[AUDIO_FEATURES] = scaler.fit_transform(self.df[AUDIO_FEATURES])

        return self
    
    def dataset_summary(self):

        valid_years = self.df[
            self.df["year"].notna() &
            (self.df["year"] != 0)
        ]

        return {
            "Songs": len(self.df),
            "Artists": self.df["artists"].nunique(),
            "Years": valid_years["year"].nunique(),
            "Average Popularity": round(self.df["popularity"].dropna().mean(),1),
            "Genres": "To be computed later"
        }
    
    def missing_values(self):
        return self.df.isnull().sum()

    def duplicate_count(self):
        return self.df.duplicated().sum()

    def correlation_matrix(self):
        return self.df[AUDIO_FEATURES].corr()

    def get_dataframe(self):
        return self.df