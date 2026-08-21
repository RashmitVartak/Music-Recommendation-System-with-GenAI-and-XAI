import numpy as np


class ContentEvaluator:

    def __init__(self, recommender):
        self.recommender = recommender

        # Fast song-name → index lookup
        self.name_to_index = {}

        for index, name in enumerate(recommender.df["name"].fillna("")):
            normalized_name = str(name).lower()

            if normalized_name not in self.name_to_index:
                self.name_to_index[normalized_name] = index

    def _get_index(self, song_name):
        return self.name_to_index.get(str(song_name).lower())

    def _similarity(self, query_index, recommended_indices):
        query_vector = (
            self.recommender.normalized_feature_matrix[query_index]
        )

        vectors = (
            self.recommender.normalized_feature_matrix[recommended_indices]
        )

        return vectors @ query_vector

    def _diversity(self, recommended_indices):

        if len(recommended_indices) < 2:
            return 0.0

        vectors = (
            self.recommender.normalized_feature_matrix[recommended_indices]
        )

        similarity_matrix = vectors @ vectors.T
        n = len(vectors)
        pairwise_similarities = similarity_matrix[np.triu_indices(n, k=1)]

        if len(pairwise_similarities) == 0:
            return 0.0

        return float(1 - np.mean(pairwise_similarities))

    def evaluate(self,song_names,ks=(5, 10, 20)):
        """
        Evaluate Content-Based recommendations.

        Recommendations are generated only once at
        max(K) and reused for all K values.
        """

        max_k = max(ks)

        results = {
            k: {
                "similarity": [],
                "diversity": []
            }
            for k in ks
        }

        # Used for catalog coverage
        recommended_catalog = {
            k: set()
            for k in ks
        }

        total = len(song_names)

        for count, song_name in enumerate(song_names,start=1):

            query_index = self._get_index(song_name)

            if query_index is None:
                continue

            # Generate recommendations ONLY ONCE
            recommendations = self.recommender.recommend(song_name,n=max_k)

            if (recommendations is None or recommendations.empty):
                continue

            recommended_indices = [
                self._get_index(name)
                for name in recommendations["name"]
            ]

            recommended_ids = (recommendations["id"].tolist())

            valid_pairs = [
                (index, song_id)
                for index, song_id
                in zip(recommended_indices,recommended_ids)
                if index is not None
            ]

            if not valid_pairs:
                continue

            recommended_indices = [pair[0]for pair in valid_pairs]

            recommended_ids = [pair[1]for pair in valid_pairs]

            # Calculate all K values from the
            # same recommendation list
            for k in ks:

                indices_k = recommended_indices[:k]
                ids_k = recommended_ids[:k]

                similarities = self._similarity(query_index,indices_k)

                results[k]["similarity"].append(float(np.mean(similarities)))

                results[k]["diversity"].append(self._diversity(indices_k))

                recommended_catalog[k].update(ids_k)

            if count % 25 == 0:
                print(f"Evaluated "
                    f"{count}/{total} songs")

        final_results = {}

        catalog_size = len(self.recommender.df)

        for k in ks:

            similarities = (results[k]["similarity"])

            diversities = (results[k]["diversity"])

            coverage = (
                len(recommended_catalog[k])
                / catalog_size
                if catalog_size > 0
                else 0.0
            )

            final_results[k] = {
                "Average Similarity@K": (
                    float(np.mean(similarities))
                    if similarities
                    else 0.0
                ),
                "Intra-List Diversity@K": (
                    float(np.mean(diversities))
                    if diversities
                    else 0.0
                ),
                "Catalog Coverage@K": coverage,
                "Songs Evaluated": len(similarities)
            }

        return final_results