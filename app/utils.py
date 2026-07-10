AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

def format_number(number):

    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"

    if number >= 1_000:
        return f"{number/1_000:.1f}K"

    return str(number)