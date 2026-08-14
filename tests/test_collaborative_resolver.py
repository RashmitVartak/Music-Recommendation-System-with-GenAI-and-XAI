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

song_name="Hotel California - 2013 Remaster"
artist_name="Eagles"

song_id = collaborative.resolve_song_id(
    song_name,
    artist_name
)
# song_name = "21 Guns [feat. Green Day & The Cast Of American Idiot] (Album Version)"
# artist_name = "Green Day"

# song_id = collaborative.resolve_song_id(
#     song_name,
#     artist_name
# )

print("=" * 60)
print("Song:", song_name)
print("Artist:", artist_name)
print("Resolved song_id:", song_id)

if song_id is None:
    print("❌ Not found in song_data.csv")

else:
    print("✅ Found in song_data.csv")

    # Check whether this song actually has user interactions
    if song_id in collaborative.song_user_matrix.index:
        print("✅ Has collaborative interactions")
    else:
        print("❌ NO collaborative interactions")

    print(
        "Total songs in interaction matrix:",
        len(collaborative.song_user_matrix)
    )

    recommendations = collaborative.recommend_by_id(
        song_id,
        n=10
    )

    print("\n========== Recommendations ==========")
    print(
        recommendations[
            ["name", "artists", "score"]
        ]
    )