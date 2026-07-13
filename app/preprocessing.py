import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.utils import AUDIO_FEATURES


class SpotifyPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)

        return self

    def prepare_audio_features(self):
        scaler = StandardScaler()
        self.df[AUDIO_FEATURES] = scaler.fit_transform(self.df[AUDIO_FEATURES])

        return self
    
    def dataset_summary(self):

        return {

            "Songs": len(self.df),
            "Artists": self.df["artists"].nunique(),
            "Years": self.df["year"].nunique(),
            "Average Popularity": round(self.df["popularity"].mean(),2),
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