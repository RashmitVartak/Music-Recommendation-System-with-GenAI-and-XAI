import streamlit as st

from app.data_loader import SpotifyDataLoader
from app.preprocessing import SpotifyPreprocessor
from app.recommenders.content_based import ContentBasedRecommender
from app.utils import format_number
from app.recommenders.popularity import (PopularityRecommender)
from app.recommenders.collaborative import (CollaborativeRecommender)
from app.recommenders.hybrid import HybridRecommender

#adding cache funtions
@st.cache_resource
def get_content_recommender(songs):
    return ContentBasedRecommender(songs)


@st.cache_resource
def get_popularity_recommender(songs):
    return PopularityRecommender(songs)


@st.cache_resource
def get_collaborative_recommender():
    return CollaborativeRecommender(
        triplets_path="datasets/triplets_file.csv",
        song_data_path="datasets/song_data.csv"
    )


@st.cache_data
def load_spotify_dataset():

    loader = SpotifyDataLoader().load_data()

    processor = (
        SpotifyPreprocessor(loader.song_df)
        .clean_data()
        .prepare_audio_features()
    )

    return processor




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
processor = load_spotify_dataset()

songs = processor.get_dataframe()

summary = processor.dataset_summary()

# Initializing the recommenders 
recommender = get_content_recommender(songs)

popularity = get_popularity_recommender(songs)

collaborative = get_collaborative_recommender()

hybrid = HybridRecommender(recommender,collaborative)

# Dataset Statistics
st.subheader("📊 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎵 Songs",format_number(summary["Songs"]))
col2.metric("🎤 Artists", format_number(summary["Artists"]))
col3.metric("📅 Years", summary["Years"])
col4.metric("⭐ Avg Popularity", summary["Average Popularity"])

st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Content",
        "🔥 Popular",
        "👥 Collaborative",
        "⭐ Hybrid"
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
            # st.success(f"Top {top_n} songs similar to **{song}**")
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

with tab3:

    st.subheader("👥 Collaborative Recommendation")

    song = st.selectbox("Choose a Song",collaborative.available_songs(),key="collab_song")
    top_n = st.slider("Number of Recommendations",5,20,10,key="collab_slider")

    if st.button("Recommend",key="collab_button"):
        recommendations = collaborative.recommend(song,top_n)

        if recommendations is None:
            st.error("Song not found.")

        else:
            for _, row in recommendations.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])

                    with c1:
                        st.subheader(row["title"])
                        st.write(f"**Artist:** {row['artist_name']}")
                        st.write(f"**Year:** {row['year']}")

                    with c2:
                        st.metric("Similarity",f"{row['Similarity']*100:.1f}%")

with tab4:

    st.subheader("⭐ Hybrid Recommendation")

    song = st.selectbox("Choose Song",recommender.available_songs(),key="hybrid_song")

    top_n = st.slider("Number of Recommendations",5,20,10,key="hybrid_slider")
    content_weight = st.slider("Content Weight",
                               0.0,
                               1.0,
                               0.6,
                               0.1
                               )

    collaborative_weight = 1 - content_weight
    hybrid.content_weight = content_weight
    hybrid.collaborative_weight = collaborative_weight

    st.write(f"Content : {content_weight:.1f}")
    st.write(f"Collaborative : {collaborative_weight:.1f}")

    if st.button("Generate Hybrid Recommendations",key="hybrid_button"):
        
        recommendations = hybrid.recommend(song,top_n)
        if recommendations is None:
            st.error("No recommendations found.")

        else:
            for _, row in recommendations.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4,1])

                    with c1:
                        st.subheader(row["name"])
                        st.write(f"**Artist:** {row['artists']}")
                        st.write(f"**Year:** {row['year']}")

                    with c2:
                        st.metric("Hybrid",f"{row['Hybrid Score']:.3f}")


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

# st.markdown("---")
# st.header("🧪 Collaborative Dataset Preview")

# manager = (
#     DataManager()
#     .load_triplets("datasets/triplets_file.csv")
#     .load_song_data("datasets/song_data.csv")
#     .merge()
# )

# merged = manager.get_dataset()

# st.write("Merged Dataset Shape:", merged.shape)

# st.dataframe(
#     merged.head(10),
#     use_container_width=True
# )