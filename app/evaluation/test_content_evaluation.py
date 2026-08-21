import pandas as pd
import time

from app.recommenders.content_based import (ContentBasedRecommender)
from app.evaluation.content_evaluator import (ContentEvaluator)


DATASET_PATH = "datasets/processed/merged_dataset.csv"
SAMPLE_SIZE = 500
K_VALUES = (5, 10, 20)

# Load dataset
songs = pd.read_csv(DATASET_PATH)

# Create recommender
recommender = ContentBasedRecommender(songs)

# Create evaluator
evaluator = ContentEvaluator(recommender)

# Select reproducible sample
song_names = (songs["name"]
            .dropna()
            .drop_duplicates()
            .sample(n=SAMPLE_SIZE,random_state=42)
            .tolist())

# Evaluation
print("\n" + "=" * 60)
print("CONTENT-BASED RECOMMENDER EVALUATION")
print("=" * 60)

print(f"Songs available : "f"{songs['name'].nunique()}")

print(f"Songs selected  : "f"{len(song_names)}")

print(f"K values        : "f"{K_VALUES}")

start_time = time.time()

results = evaluator.evaluate(song_names=song_names,ks=K_VALUES)


elapsed = time.time() - start_time

# Display results
for k in K_VALUES:

    metrics = results[k]

    print("\n" + "=" * 60)
    print(f"K = {k}")
    print("=" * 60)

    print(f"Average Similarity@K     : "f"{metrics['Average Similarity@K']:.4f}")

    print(f"Intra-List Diversity@K   : "f"{metrics['Intra-List Diversity@K']:.4f}")

    print(f"Catalog Coverage@K       : "f"{metrics['Catalog Coverage@K'] * 100:.2f}%")

    print(f"Songs Evaluated          : "f"{metrics['Songs Evaluated']}")


print("\n" + "=" * 60)

print(f"TOTAL EXECUTION TIME : "
      f"{elapsed:.2f} seconds")

print("=" * 60)