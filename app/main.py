import streamlit as st

from data_loader import SpotifyDataLoader

loader = SpotifyDataLoader().load_data()

st.title("Spotify Music Recommendation System")

st.write(loader.song_df.head())