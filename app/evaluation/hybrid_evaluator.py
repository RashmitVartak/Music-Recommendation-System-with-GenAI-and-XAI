import numpy as np


class HybridEvaluator:

    def __init__(self, recommender):
        self.recommender = recommender

        # Fast song-name -> index lookup
        self.name_to_index = {}

        for index, name in enumerate(recommender.content.df["name"].fillna("")):
            normalized_name = str(name).lower()

            if normalized_name not in self.name_to_index:
                self.name_to_index[normalized_name] = index

    def _get_index(self, song_name):
        return self.name_to_index.get(str(song_name).lower())

    def _similarity(self,query_index,recommended_indices):
        query_vector = (
            self.recommender.content.normalized_feature_matrix[query_index]
        )

        vectors = (
            self.recommender.content.normalized_feature_matrix[recommended_indices]
        )

        return vectors @ query_vector

    def _diversity(self, recommended_indices):

        if len(recommended_indices) < 2:
            return 0.0

        vectors = (
            self.recommender.content.normalized_feature_matrix[recommended_indices]
        )

        similarity_matrix = vectors @ vectors.T

        n = len(vectors)

        pairwise_similarities = similarity_matrix[np.triu_indices(n, k=1)]

        if len(pairwise_similarities) == 0:
            return 0.0

        return float(1 - np.mean(pairwise_similarities))

    def evaluate(self,song_names,ks=(5, 10, 20)):
        """
        Evaluate Hybrid recommendations.

        Recommendations are generated once at max(K)
        and reused for all K values.

        Metrics:
        - Average Similarity@K
        - Intra-List Diversity@K
        - Catalog Coverage@K
        """

        max_k = max(ks)

        results = {
            k: {
                "similarity": [],
                "diversity": [],
                "recommended_songs": set()
            }
            for k in ks
        }

        total = len(song_names)

        for count, song_name in enumerate(song_names,start=1):

            query_index = self._get_index(song_name)

            if query_index is None:
                continue

            # Generate Hybrid recommendations ONCE
            recommendations = (self.recommender.recommend(song_name,top_n=max_k))

            if (recommendations is None or recommendations.empty):
                continue

            # Resolve recommendation names
            recommended_indices = []
            recommended_ids = []

            for _, row in recommendations.iterrows():
                index = self._get_index(row.get("name", ""))

                if index is not None:
                    recommended_indices.append(index)
                    recommended_ids.append(row.get("id"))

            if not recommended_indices:
                continue

            # Evaluate all K values
            for k in ks:

                indices_k = (recommended_indices[:k])
                ids_k = (recommended_ids[:k])

                # Similarity
                similarities = self._similarity(query_index,indices_k)

                results[k]["similarity"].append(
                    float(np.mean(similarities))
                )

                # Diversity
                results[k]["diversity"].append(
                    self._diversity(indices_k))

                # Coverage
                results[k]["recommended_songs"].update(song_id
                                                    for song_id in ids_k
                                                    if song_id is not None)

            if count % 25 == 0:
                print(f"Evaluated "f"{count}/{total} songs")

        # Final results
        catalog_size = len(self.recommender.content.df)
        final_results = {}

        for k in ks:

            similarities = (results[k]["similarity"])
            diversities = (results[k]["diversity"])

            coverage = (
                len(results[k]["recommended_songs"])
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