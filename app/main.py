import streamlit as st

from app.data_loader import SpotifyDataLoader
from app.preprocessing import SpotifyPreprocessor
from app.recommenders.content_based import ContentBasedRecommender
from app.utils import format_number
from app.recommenders.popularity import (PopularityRecommender)



# Page Configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.success("Phase 2 - Content-Based Recommendation")

# Title
st.title("Music Recommendation System with GenAI & XAI")
st.write(
    "A modern Music Recommendation System using Content-Based Filtering, "
    "Collaborative Filtering, Explainable AI (XAI) and Generative AI."
)

# Load Dataset
loader = SpotifyDataLoader().load_data()

processor = (
    SpotifyPreprocessor(loader.song_df)
    .clean_data()
    .prepare_audio_features()
)

songs = processor.get_dataframe()
summary = processor.dataset_summary()

# Initialize Recommendation Engine
recommender = ContentBasedRecommender(songs)

#initialize Popularity Recommender
popularity = PopularityRecommender(songs)


# Dataset Statistics
st.subheader("📊 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎵 Songs",format_number(summary["Songs"]))
col2.metric("🎤 Artists", format_number(summary["Artists"]))
col3.metric("📅 Years", summary["Years"])
col4.metric("⭐ Avg Popularity", summary["Average Popularity"])

st.markdown("---")


tab1, tab2 = st.tabs(
    [
        "🎯 Content Based",
        "🔥 Popular Songs"
    ]
)
# ----------------------------------------------------------
# Content-Based Recommendation
# ----------------------------------------------------------
with tab1:
    st.subheader("Content-Based Recommendation")
    st.markdown("Find Similar Songs")

    col1, col2 = st.columns([3,1])

    with col1:
        song = st.selectbox("Choose a Song",recommender.available_songs())

    with col2:
        top_n = st.number_input("Top",min_value=5,max_value=20,value=10)

    if st.button("Recommend Songs"):

        recommendations = recommender.recommend(song_name=song,n=top_n)

        if recommendations is None:

            st.error("Song not found.")

        else:

            st.success(f"Top {top_n} songs similar to **{song}**")

            # st.dataframe(
            #     recommendations,
            #     use_container_width=True
            # )

            st.markdown("## Recommendations")

            for _, row in recommendations.iterrows():

                with st.container(border=True):
                    c1, c2 = st.columns([2,1])

                    with c1:
                        st.subheader(row["name"])
                        st.write(f"**Artist:** {row['artists']}")
                        st.write(f"**Year:** {row['year']}")

                    with c2:
                        st.metric("Similarity", f"{row['Similarity Score']*100:.1f}%")
                        st.metric("Popularity", row["popularity"])

    st.markdown("---")

with tab2:

    st.subheader("🔥 Most Popular Songs")

    top_n = st.slider("Top Songs",5,20,10,key="popular_slider")
    popular = popularity.recommend(top_n)

    for _, row in popular.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([4,1])
            with c1:
                st.subheader(row["name"])
                st.write(f"**Artist:** {row['artists']}")

                st.write(f"**Year:** {row['year']}")

            with c2:
                st.metric("Popularity",row["popularity"])


# Dataset Preview
# st.subheader("🎼 Dataset Preview")
# st.dataframe(
#     songs.head(15),
#     use_container_width=True
# )
# st.markdown("---")

# Correlation Matrix
# st.subheader("📈 Audio Feature Correlation")
# st.dataframe(
#     processor.correlation_matrix(),
#     use_container_width=True
# )
# st.markdown("---")

# Missing Values
# st.subheader("🧹 Missing Values")
# st.dataframe(
#     processor.missing_values(),
#     use_container_width=True
# )