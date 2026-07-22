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


import streamlit as st

def diversity_card(title, score, insight, emoji):

    with st.container(border=True):

        st.markdown(f"### {emoji} {title}")

        st.metric(
            label="",
            value=f"{score:.0f}%"
        )

        st.progress(score / 100)

        st.caption(insight)