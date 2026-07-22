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

    # Dynamic percentage color
    if score >= 80:
        color = "#22C55E"      # Green
    elif score >= 60:
        color = "#3B82F6"      # Blue
    elif score >= 40:
        color = "#F59E0B"      # Orange
    else:
        color = "#EF4444"      # Red

    st.markdown(
        f"""
        <div style="
            border:1px solid #3A3A3A;
            border-radius:12px;
            padding:12px 16px;
            background:#000000;
            min-height:120px;
        ">

            <h4 style="margin-bottom:18px;">
                {emoji} {title}
            </h4>

            <h1 style="
                color:{color};
                margin:0;
                font-size:42px;
                font-weight:700;
            ">
                {score:.0f}%
            </h1>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(score / 100)

    st.caption(insight)