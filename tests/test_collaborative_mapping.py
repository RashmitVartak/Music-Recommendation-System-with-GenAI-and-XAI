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
    ("Hotel California", "Eagles"),
    ("Hotel California - 2013 Remaster", "Eagles"),
    ("Hotel California - Live; 1999 Remaster", "Eagles"),

    ("Bohemian Rhapsody", "Queen"),
    ("Bohemian Rhapsody - Remastered", "Queen"),

    ("Sweet Child O' Mine", "Guns N' Roses"),
    ("Sweet Child O' Mine - Live", "Guns N' Roses"),
]


for title, artist in test_cases:
    song_id = collaborative.resolve_song_id(title,artist)

    print("=" * 60)
    print("Title :", title)
    print("Artist:", artist)
    print("Resolved song_id:", song_id)

    if song_id:
        print("✅ FOUND")
    else:
        print("❌ NOT FOUND")