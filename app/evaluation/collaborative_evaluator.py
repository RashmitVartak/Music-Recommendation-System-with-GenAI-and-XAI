import pandas as pd
import numpy as np
import time


class CollaborativeEvaluator:

    def __init__(self, triplets_path):
        self.triplets = pd.read_csv(triplets_path)

    # Ranking Metrics
    def precision_at_k(self, recommended, actual, k=10):
        recommended = recommended[:k]

        if not recommended:
            return 0.0

        hits = len(set(recommended) & set(actual))

        return hits / len(recommended)

    def recall_at_k(self, recommended, actual, k=10):

        if not actual:
            return 0.0

        recommended = recommended[:k]
        hits = len(set(recommended) & set(actual))

        return hits / len(actual)

    def hit_rate_at_k(self, recommended, actual, k=10):

        recommended = recommended[:k]
        return float(bool(set(recommended) & set(actual)))

    def ndcg_at_k(self, recommended, actual, k=10):

        recommended = recommended[:k]

        if not actual:
            return 0.0

        relevance = [
            1 if song in actual else 0
            for song in recommended
        ]

        dcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(relevance)
        )

        ideal_hits = min(len(actual), k)

        idcg = sum(
            1 / np.log2(i + 2)
            for i in range(ideal_hits)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # Evaluation
    def evaluate(self,recommender,k=10,min_songs=3,max_users=500,random_state=42):
        start_time = time.time()

        # Songs listened by each user
        user_songs = (self.triplets
                    .groupby("user_id")["song_id"]
                    .apply(list))

        # Keep users with enough interactions
        eligible_users = user_songs[user_songs.apply(len) >= min_songs]

        print(f"Eligible users: {len(eligible_users)}")

        # Sample users
        if len(eligible_users) > max_users:
            eligible_users = eligible_users.sample(n=max_users,random_state=random_state)

        print(f"Users selected: {len(eligible_users)}")
        print("-" * 60)

        results = []

        total_users = len(eligible_users)

        for count, (user_id, songs) in enumerate(eligible_users.items(),start=1):

            # First song = seed
            seed_song = songs[0]
            # Remaining songs = ground truth
            actual_songs = set(songs[1:])

            recommendations = recommender.recommend_by_id(seed_song,n=k)

            if (recommendations is None or recommendations.empty):
                continue

            recommended_songs = (recommendations["id"].tolist())

            results.append({
                "user_id": user_id,
                "seed_song": seed_song,
                "precision": self.precision_at_k(recommended_songs,actual_songs,k),
                "recall": self.recall_at_k(recommended_songs,actual_songs,k),
                "hit_rate": self.hit_rate_at_k(recommended_songs,actual_songs,k),
                "ndcg": self.ndcg_at_k(recommended_songs,actual_songs,k)
            })

            # Progress
            if count % 25 == 0 or count == total_users:
                elapsed = time.time() - start_time

                print(
                    f"Evaluated {count}/{total_users} users "
                    f"({count / total_users * 100:.1f}%) "
                    f"| Time: {elapsed:.1f}s"
                )

        if not results:
            return None

        results = pd.DataFrame(results)

        elapsed = time.time() - start_time

        return {
            "Precision@K": results["precision"].mean(),
            "Recall@K": results["recall"].mean(),
            "Hit Rate@K": results["hit_rate"].mean(),
            "NDCG@K": results["ndcg"].mean(),
            "Users Evaluated": len(results),
            "Execution Time (seconds)": round(elapsed, 2)
        }