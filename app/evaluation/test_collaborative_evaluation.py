from recommenders.collaborative import CollaborativeRecommender
from evaluation.collaborative_evaluator import CollaborativeEvaluator


TRIPLETS_PATH = "../datasets/triplets_file.csv"
SONG_DATA_PATH = "../datasets/song_data.csv"


collaborative = CollaborativeRecommender(triplets_path=TRIPLETS_PATH,
                                        song_data_path=SONG_DATA_PATH)

evaluator = CollaborativeEvaluator(triplets_path=TRIPLETS_PATH)

print("\n" + "=" * 60)
print("COLLABORATIVE RECOMMENDER EVALUATION")
print("=" * 60)

for k in [5, 10, 20]:

    print(f"\n{'=' * 60}")
    print(f"EVALUATING K = {k}")
    print(f"{'=' * 60}")

    results = evaluator.evaluate(recommender=collaborative,
                                k=k,
                                min_songs=3,
                                max_users=500,
                                random_state=42)

    if results is None:
        print("No users available for evaluation.")
        continue

    print("\nResults:")
    print("-" * 40)

    for metric, value in results.items():

        if metric in ["Users Evaluated"]:
            print(f"{metric:<25}: {value}")

        elif metric == "Execution Time (seconds)":
            print(f"{metric:<25}: {value:.2f}")

        else:
            print(f"{metric:<25}: {value:.4f}")