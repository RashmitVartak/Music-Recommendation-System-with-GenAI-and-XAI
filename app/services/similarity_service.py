from sklearn.metrics.pairwise import cosine_similarity


class SimilarityService:

    def build_similarity_matrix(self, feature_matrix):

        return cosine_similarity(feature_matrix)