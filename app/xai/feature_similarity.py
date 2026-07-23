import pandas as pd


class FeatureSimilarity:

    AUDIO_FEATURES = [
        "danceability",
        "energy",
        "valence",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "tempo"
    ]

    @staticmethod
    def compare_features(song1, song2):
        """
        Compare audio features of two songs.

        Returns
        -------
        DataFrame
            Feature | Song1 | Song2 | Difference
        """

        comparisons = []

        for feature in FeatureSimilarity.AUDIO_FEATURES:
            if feature not in song1 or feature not in song2:
                continue

            value1 = float(song1[feature])
            value2 = float(song2[feature])

            comparisons.append({
                "Feature": feature,
                "Song 1": round(value1, 3),
                "Song 2": round(value2, 3),
                "Difference": round(abs(value1 - value2), 3)
            })

        #DEBUG
        # print("Song1 keys:", song1.index.tolist())
        # print("Song2 keys:", song2.index.tolist())
        # print("Comparisons:", comparisons)

        comparison_df = pd.DataFrame(comparisons)
        comparison_df = comparison_df.sort_values("Difference")

        return comparison_df.reset_index(drop=True)
    

    @staticmethod
    def top_matching_features(song1, song2, top_n=5):
        """
        Return the most similar audio features.
        """
        comparison = FeatureSimilarity.compare_features(song1,song2)
        return comparison.head(top_n)


    @staticmethod
    def similarity_score(song1, song2):

        comparison = FeatureSimilarity.compare_features(song1,song2)

        if comparison.empty:
            return 0

        avg_difference = comparison["Difference"].mean()
        similarity = max(0,100 * (1 - avg_difference))

        return round(similarity, 1)
    
    @staticmethod
    def matching_feature_names(song1, song2, top_n=5):
        """
        Return only the names of the top matching features.
        """
        comparison = FeatureSimilarity.top_matching_features(song1,song2,top_n)

        return [
            feature.replace("_", " ").title()
            for feature in comparison["Feature"]
        ]
    
    @staticmethod
    def feature_similarity_scores(song1, song2, top_n=5):
        """
        Return the top matching features along with
        their similarity percentages.
        """

        comparison = FeatureSimilarity.top_matching_features(song1,song2,top_n)

        similarities = []

        for _, row in comparison.iterrows():

            feature = row["Feature"]
            difference = row["Difference"]

            if feature == "tempo":
                similarity = max(0, 100 - (difference / 200) * 100)
            else:
                similarity = max(0, 100 - (difference * 100))

            similarities.append({
                "feature": feature.replace("_", " ").title(),
                "similarity": round(similarity, 1)
            })

        return similarities