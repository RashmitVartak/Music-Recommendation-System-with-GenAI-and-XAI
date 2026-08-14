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


# test_songs = ["Believer",
#     "Hotel California",]

test_cases = [
    ("21 Guns [feat. Green Day & The Cast Of American Idiot] (Album Version)", "Green Day"),
    ("Hotel California", "Eagles"),
    ("Believer", "Imagine Dragons"),
]

for song_name, artist_name in test_cases:

    song_id = collaborative.resolve_song_id(
        song_name,
        artist_name
    )

    has_interactions = collaborative.has_interactions(song_id)

    print("=" * 60)
    print("Song:", song_name)
    print("Artist:", artist_name)
    print("Resolved song_id:", song_id)
    print("Has interactions:", has_interactions)