import pandas as pd


# Collaborative results
collaborative_results = pd.DataFrame([
    {
        "Model": "Collaborative",
        "K": 5,
        "Precision": 0.1128,
        "Recall": 0.0384,
        "Hit Rate": 0.2860,
        "NDCG": 0.1212,
        "Similarity": None,
        "Diversity": None,
        "Coverage": None
    },
    {
        "Model": "Collaborative",
        "K": 10,
        "Precision": 0.0974,
        "Recall": 0.0589,
        "Hit Rate": 0.3820,
        "NDCG": 0.1087,
        "Similarity": None,
        "Diversity": None,
        "Coverage": None
    },
    {
        "Model": "Collaborative",
        "K": 20,
        "Precision": 0.0739,
        "Recall": 0.0840,
        "Hit Rate": 0.4740,
        "NDCG": 0.0936,
        "Similarity": None,
        "Diversity": None,
        "Coverage": None
    }
])


# Content results
content_results = pd.DataFrame([
    {
        "Model": "Content-Based",
        "K": 5,
        "Precision": None,
        "Recall": None,
        "Hit Rate": None,
        "NDCG": None,
        "Similarity": 0.8354,
        "Diversity": 0.2965,
        "Coverage": 0.0079
    },
    {
        "Model": "Content-Based",
        "K": 10,
        "Precision": None,
        "Recall": None,
        "Hit Rate": None,
        "NDCG": None,
        "Similarity": 0.8376,
        "Diversity": 0.2876,
        "Coverage": 0.0158
    },
    {
        "Model": "Content-Based",
        "K": 20,
        "Precision": None,
        "Recall": None,
        "Hit Rate": None,
        "NDCG": None,
        "Similarity": 0.8353,
        "Diversity": 0.2903,
        "Coverage": 0.0314
    }
])

# Hybrid results — selected 60/40 configuration
hybrid_results = pd.DataFrame([
    {
        "Model": "Hybrid (60/40)",
        "K": 5,
        "Precision": None,
        "Recall": None,
        "Hit Rate": None,
        "NDCG": None,
        "Similarity": 0.8357,
        "Diversity": 0.2961,
        "Coverage": 0.0079
    },
    {
        "Model": "Hybrid (60/40)",
        "K": 10,
        "Precision": None,
        "Recall": None,
        "Hit Rate": None,
        "NDCG": None,
        "Similarity": 0.8376,
        "Diversity": 0.2876,
        "Coverage": 0.0158
    },
    {
        "Model": "Hybrid (60/40)",
        "K": 20,
        "Precision": None,
        "Recall": None,
        "Hit Rate": None,
        "NDCG": None,
        "Similarity": 0.8353,
        "Diversity": 0.2904,
        "Coverage": 0.0314
    }
])


# Combine
summary = pd.concat(
    [
        collaborative_results,
        content_results,
        hybrid_results
    ],
    ignore_index=True
)

# Save complete summary
OUTPUT_PATH = ("app/evaluation/evaluation_summary.csv")
summary.to_csv(OUTPUT_PATH,index=False)


# Display

print("\n" + "=" * 80)
print("FINAL RECOMMENDER EVALUATION SUMMARY")
print("=" * 80)

display_summary = summary.copy()

display_summary["Coverage"] = (display_summary["Coverage"] * 100)

print(
    display_summary.to_string(
        index=False,
        formatters={
            "Precision":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.4f}",

            "Recall":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.4f}",

            "Hit Rate":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.4f}",

            "NDCG":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.4f}",

            "Similarity":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.4f}",

            "Diversity":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.4f}",

            "Coverage":
                lambda x:
                "-" if pd.isna(x)
                else f"{x:.2f}%"
        }
    )
)

print("\n" + "=" * 80)
print(
    f"Saved to: {OUTPUT_PATH}"
)
print("=" * 80)