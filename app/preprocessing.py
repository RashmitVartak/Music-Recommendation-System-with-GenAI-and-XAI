import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import AUDIO_FEATURES


class SpotifyPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):

        self.df.drop_duplicates(inplace=True)

        self.df.dropna(inplace=True)

        return self

    def prepare_audio_features(self):

        scaler = StandardScaler()

        self.df[AUDIO_FEATURES] = scaler.fit_transform(
            self.df[AUDIO_FEATURES]
        )

        return self

    def get_dataframe(self):

        return self.df
    
    def dataset_summary(self):

        return {

            "Songs": len(self.df),

            "Artists": self.df["artists"].nunique(),

            "Years": self.df["year"].nunique(),

            "Genres": "To be computed later"

        }