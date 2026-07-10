import streamlit as st

from preprocessing import SpotifyPreprocessor
from data_loader import SpotifyDataLoader
from recommenders.content_based import ContentBasedRecommender

#page configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

#nav bar
st.sidebar.title("Navigation")
st.sidebar.success("Phase 1 - Data Exploration")


st.title("Spotify Music Recommendation System")


loader = SpotifyDataLoader().load_data()
processor = (
    SpotifyPreprocessor(loader.song_df).clean_data()
    )

summary = processor.dataset_summary()

recommender = (
    ContentBasedRecommender(
        processor.get_dataframe()
    )
    .build_similarity_matrix()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Songs",summary["Songs"])
col2.metric("Artists",summary["Artists"])
col3.metric("Years",summary["Years"])
col4.metric("Popularity",summary["Average Popularity"])

#displays missing values
st.subheader("Missing Values")

st.dataframe(
    processor.missing_values()
)

#displays dataset
st.subheader("Dataset Preview")

st.dataframe(
    processor.get_dataframe().head(15)
)

#displays correlation matrix
st.subheader("Feature Correlation")

st.dataframe(
    processor.correlation_matrix()
)

st.subheader("Testing Recommendation Engine")

recommendations = recommender.recommend(
    "Shape of You"
)

st.dataframe(recommendations)
