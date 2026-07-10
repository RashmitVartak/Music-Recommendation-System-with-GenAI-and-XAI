import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.utils import AUDIO_FEATURES


class FeatureService:

    def __init__(self):
        self.scaler = StandardScaler()

    def create_feature_matrix(self, df: pd.DataFrame):
        features = df[AUDIO_FEATURES].copy()
        features = self.scaler.fit_transform(features)
        return features

    def fit_scaler(self, df):
        self.scaler.fit(df[AUDIO_FEATURES])

    def transform(self, df):
        return self.scaler.transform(df[AUDIO_FEATURES])