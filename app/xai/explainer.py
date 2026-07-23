from app.xai.feature_similarity import FeatureSimilarity


class RecommendationExplainer:

    @staticmethod
    def explain_content(selected_song, recommended_song):
        """
        Generate explanation for content-based recommendations.
        """

        similarity = FeatureSimilarity.similarity_score(selected_song,recommended_song)

        features = FeatureSimilarity.feature_similarity_scores(selected_song,recommended_song)
        feature_names = [feature["feature"] for feature in features]
        
        if len(feature_names) > 1:
            explanation = (
                f"This song was recommended because it closely matches "
                f"the selected song in "
                f"{', '.join(feature_names[:-1])} "
                f"and {feature_names[-1]}."
            )

        else:
            explanation = (
                f"This song was recommended because of its similar "
                f"{feature_names[0]}."
            )

        return {
            "similarity_score": similarity,
            "matching_features": features,
            "explanation": explanation
        }

    @staticmethod
    def explain_popularity(recommended_song):
        """
        Generate explanation for popularity-based recommendations.
        """

        popularity = recommended_song.get("popularity", "N/A")

        explanation = (
            "This song was recommended because it is among the "
            "most popular tracks in the dataset."
        )

        return {
            "popularity": popularity,
            "explanation": explanation
        }

    @staticmethod
    def explain_collaborative():
        """
        Generate explanation for collaborative recommendations.
        """

        explanation = (
            "This song was recommended because users with similar "
            "listening preferences also enjoyed it."
        )

        return {
            "explanation": explanation
        }

    @staticmethod
    def explain_hybrid(selected_song, recommended_song):
        """
        Generate explanation for hybrid recommendations.
        """
        content = RecommendationExplainer.explain_content(selected_song,recommended_song)
        explanation = (
            "This recommendation combines musical similarity "
            "with collaborative listening patterns."
        )

        return {
            "similarity_score": content["similarity_score"],
            "matching_features": content["matching_features"],
            # "feature_similarity_scores":content["feature_similarity_scores"],
            "explanation": explanation
        }
    