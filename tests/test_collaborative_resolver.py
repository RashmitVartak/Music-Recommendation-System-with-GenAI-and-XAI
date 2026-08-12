import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from app.recommenders.collaborative import CollaborativeRecommender


collaborative = CollaborativeRecommender(
    triplets_path="datasets/triplets_file.csv",
    song_data_path="datasets/song_data.csv"
)


song_id = collaborative.resolve_song_id(
    "Hotel California - 2013 Remaster",
    "Eagles"
)

print("=" * 60)
print("Resolved song_id:", song_id)

if song_id is None:
    print("❌ Could not resolve song")
else:
    print("✅ Song resolved")

    recommendations = collaborative.recommend(
        "Hotel California",
        n=10
    )

    print("\n========== Collaborative Recommendations ==========")
    print(
        recommendations[
            [
                "name",
                "artists",
                "score"
            ]
        ]
    )