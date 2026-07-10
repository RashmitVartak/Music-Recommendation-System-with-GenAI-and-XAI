from app.utils import AUDIO_FEATURES

class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()
    
    def select_audio_features(self):
        self.df = self.df[AUDIO_FEATURES]

        return self