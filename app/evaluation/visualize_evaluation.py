import os
import pandas as pd
import matplotlib.pyplot as plt


# PATHS

EVALUATION_PATH = ("app/evaluation/evaluation_summary.csv")
WEIGHT_RESULTS_PATH = ("app/evaluation/hybrid_weight_results.csv")
PLOTS_DIR = ("app/evaluation/plots")

os.makedirs(PLOTS_DIR,exist_ok=True)

# LOAD DATA
evaluation = pd.read_csv(EVALUATION_PATH)
weight_results = pd.read_csv(WEIGHT_RESULTS_PATH)

# 1. COLLABORATIVE METRICS
collaborative = evaluation[evaluation["Model"] == "Collaborative"].copy()
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(collaborative["K"],
    collaborative["Precision"],
    marker="o",
    label="Precision@K"
)

ax.plot(collaborative["K"],
    collaborative["Recall"],
    marker="o",
    label="Recall@K"
)

ax.plot(collaborative["K"],
    collaborative["Hit Rate"],
    marker="o",
    label="Hit Rate@K"
)

ax.plot(collaborative["K"],
    collaborative["NDCG"],
    marker="o",
    label="NDCG@K"
)

ax.set_title("Collaborative Recommender Performance")
ax.set_xlabel("K")
ax.set_ylabel("Score")
ax.set_xticks(collaborative["K"])
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR,"collaborative_metrics.png"),
    dpi=150
)

plt.close()


# 2. CONTENT vs HYBRID SIMILARITY
comparison = evaluation[
    evaluation["Model"].isin(
        ["Content-Based","Hybrid (60/40)"]
    )
].copy()

fig, ax = plt.subplots(figsize=(9, 6))

for model in comparison["Model"].unique():

    data = comparison[comparison["Model"] == model]

    ax.plot(data["K"],
        data["Similarity"],
        marker="o",
        label=model
    )

ax.set_title("Content vs Hybrid — Average Similarity")
ax.set_xlabel("K")
ax.set_ylabel("Average Similarity@K")
ax.set_xticks([5, 10, 20])
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR,"similarity_comparison.png"),
    dpi=150
)
plt.close()


# 3. CONTENT vs HYBRID DIVERSITY
fig, ax = plt.subplots(figsize=(9, 6))

for model in comparison["Model"].unique():

    data = comparison[comparison["Model"] == model]
    ax.plot(data["K"],
            data["Diversity"],
            marker="o",
            label=model
        )

ax.set_title("Content vs Hybrid — Recommendation Diversity")
ax.set_xlabel("K")
ax.set_ylabel("Intra-List Diversity@K")
ax.set_xticks([5, 10, 20])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR,"diversity_comparison.png"),
    dpi=150
)

plt.close()

# 4. CONTENT vs HYBRID COVERAGE

fig, ax = plt.subplots(figsize=(9, 6))

for model in comparison["Model"].unique():

    data = comparison[comparison["Model"] == model]
    ax.plot(data["K"],
            data["Coverage"] * 100,
            marker="o",
            label=model
        )

ax.set_title("Content vs Hybrid — Catalog Coverage")
ax.set_xlabel("K")
ax.set_ylabel("Catalog Coverage@K (%)")
ax.set_xticks([5, 10, 20])
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR,"coverage_comparison.png"),
    dpi=150
)

plt.close()


# 5. HYBRID WEIGHT SENSITIVITY — SIMILARITY

weight_results["Configuration"] = (weight_results["Content Weight"].astype(str)
                                    + " / "
                                    + weight_results["Collaborative Weight"].astype(str)
                                )


fig, ax = plt.subplots(figsize=(10, 6))

for config in weight_results["Configuration"].unique():

    data = weight_results[weight_results["Configuration"] == config]
    ax.plot(data["K"],
            data["Average Similarity"],
            marker="o",
            label=config
        )

ax.set_title("Hybrid Weight Sensitivity — Similarity")
ax.set_xlabel("K")
ax.set_ylabel("Average Similarity@K")
ax.set_xticks([5, 10, 20])
ax.legend(title="Content / Collaborative")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR,"hybrid_weight_similarity.png"),
    dpi=150
)

plt.close()


# 6. HYBRID WEIGHT SENSITIVITY — DIVERSITY
fig, ax = plt.subplots(
    figsize=(10, 6)
)

for config in weight_results["Configuration"].unique():

    data = weight_results[weight_results["Configuration"] == config]
    ax.plot(data["K"],
            data["Intra-List Diversity"],
            marker="o",
            label=config)

ax.set_title("Hybrid Weight Sensitivity — Diversity")
ax.set_xlabel("K")
ax.set_ylabel("Intra-List Diversity@K")
ax.set_xticks([5, 10, 20])
ax.legend(title="Content / Collaborative")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR,"hybrid_weight_diversity.png"),
    dpi=150
)

plt.close()

# 7. PRINT SUMMARY
print("\n" + "=" * 60)
print("EVALUATION VISUALIZATION COMPLETE")
print("=" * 60)

print(f"\nPlots saved to:\n{PLOTS_DIR}")

print("\nGenerated files:")

for filename in sorted(os.listdir(PLOTS_DIR)):

    if filename.endswith(".png"):
        print(f"  - {filename}")

print("\n" + "=" * 60)