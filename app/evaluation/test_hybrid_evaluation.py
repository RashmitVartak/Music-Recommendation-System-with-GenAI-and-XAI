import pandas as pd
import time

from app.recommenders.content_based import (ContentBasedRecommender)
from app.recommenders.collaborative import (CollaborativeRecommender)
from app.recommenders.hybrid import (HybridRecommender)
from app.evaluation.hybrid_evaluator import (HybridEvaluator)

# Paths
TRIPLETS_PATH = "datasets/triplets_file.csv"
SONG_DATA_PATH = "datasets/song_data.csv"
CONTENT_DATA_PATH = "datasets/processed/merged_dataset.csv"

# Evaluation settings
SAMPLE_SIZE = 500
K_VALUES = (5, 10, 20)

WEIGHT_CONFIGS = [
    (0.8, 0.2),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.2, 0.8)
]

# Load Content dataset
songs = pd.read_csv(CONTENT_DATA_PATH)

# Create recommenders
content = ContentBasedRecommender(songs)
collaborative = CollaborativeRecommender(triplets_path=TRIPLETS_PATH,song_data_path=SONG_DATA_PATH)

hybrid = HybridRecommender(content_recommender=content,
                            collaborative_recommender=collaborative,
                            content_weight=0.6,
                            collaborative_weight=0.4)


# Create evaluator
evaluator = HybridEvaluator(recommender=hybrid)


# Select reproducible sample
song_names = (songs["name"]
            .dropna()
            .drop_duplicates()
            .sample(n=SAMPLE_SIZE,random_state=42)
            .tolist()
        )


# Evaluation
print("\n" + "=" * 60)
print("HYBRID RECOMMENDER EVALUATION with WEIGHT TUNING")
print("=" * 60)

print(f"Songs available : "f"{songs['name'].nunique()}")
print(f"Songs selected  : "f"{len(song_names)}")
print(f"K values        : "f"{K_VALUES}")
print(f"Weight configs  : {len(WEIGHT_CONFIGS)}")

# Store all results
all_results = []

# Evaluate each weight configuration
for content_weight, collaborative_weight in WEIGHT_CONFIGS:

    print("\n" + "=" * 70)
    print(f"CONTENT = {content_weight:.1f} | "f"COLLABORATIVE = {collaborative_weight:.1f}")
    print("=" * 70)


    hybrid = HybridRecommender(content_recommender=content,
                                collaborative_recommender=collaborative,
                                content_weight=content_weight,
                                collaborative_weight=collaborative_weight
                            )


    evaluator = HybridEvaluator(recommender=hybrid)

    start_time = time.time()

    results = evaluator.evaluate(song_names=song_names,ks=K_VALUES)

    elapsed = time.time() - start_time


    # Store results
    for k in K_VALUES:

        metrics = results[k]

        row = {
            "Content Weight": content_weight,
            "Collaborative Weight": collaborative_weight,
            "K": k,
            "Average Similarity": (metrics["Average Similarity@K"]),
            "Intra-List Diversity": (metrics["Intra-List Diversity@K"]),
            "Catalog Coverage": (metrics["Catalog Coverage@K"]),
            "Songs Evaluated": (metrics["Songs Evaluated"]),
            "Execution Time": elapsed
        }

        all_results.append(row)


        print(f"\nK = {k}")
        print(f"Similarity : "f"{metrics['Average Similarity@K']:.4f}")
        print(f"Diversity  : "f"{metrics['Intra-List Diversity@K']:.4f}")
        print(f"Coverage   : "f"{metrics['Catalog Coverage@K'] * 100:.2f}%")


# Create comparison table
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("WEIGHT TUNING SUMMARY")
print("=" * 70)


summary = results_df[
    [
        "Content Weight",
        "Collaborative Weight",
        "K",
        "Average Similarity",
        "Intra-List Diversity",
        "Catalog Coverage"
    ]
].copy()


summary["Catalog Coverage"] *= 100


print(
    summary.to_string(
        index=False,
        formatters={
            "Average Similarity":"{:.4f}".format,
            "Intra-List Diversity":"{:.4f}".format,
            "Catalog Coverage":"{:.2f}%".format
        }
    )
)

# Save results
OUTPUT_PATH = ("app/evaluation/hybrid_weight_results.csv")

results_df.to_csv(OUTPUT_PATH,index=False)


print("\n" + "=" * 70)
print(f"Results saved to: {OUTPUT_PATH}")
print("=" * 70)