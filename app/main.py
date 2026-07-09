import streamlit as st
from preprocessing import SpotifyPreprocessor
from data_loader import SpotifyDataLoader

loader = SpotifyDataLoader().load_data()

songs = (
    SpotifyPreprocessor(loader.song_df)
    .clean_data()
    .prepare_audio_features()
    .get_dataframe()
)

summary = (
    SpotifyPreprocessor(loader.song_df)
    .clean_data()
    .dataset_summary()
)

st.write(summary)

st.dataframe(songs.head())

st.title("Spotify Music Recommendation System")

st.write(loader.song_df.head())