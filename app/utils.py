import pandas as pd
import streamlit as st

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


def diversity_card(title, score, insight, emoji):

    with st.container(border=True):

        st.markdown(f"### {emoji} {title}")
        st.metric(label="",value=f"{score:.0f}%")
        st.progress(score / 100)
        st.caption(insight)



def format_year(year):
    """Format release year for display."""
    if pd.isna(year) or year in [0, "0", ""]:
        return "Unknown"

    try:
        return str(int(float(year)))
    except (ValueError, TypeError):
        return "Unknown"


def format_text(value):
    """Format text fields like album, artist, genre."""
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if value == "":
        return "Unknown"

    return value

def format_duration(duration_ms):

    if pd.isna(duration_ms):
        return ""

    try:
        total_seconds = int(duration_ms / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds:02d}s"

    except Exception:
        return ""
    