import streamlit as st
import pandas as pd

from app.data_loader import SpotifyDataLoader
from app.preprocessing import SpotifyPreprocessor
from app.recommenders.content_based import ContentBasedRecommender
from app.utils import (format_number , diversity_card, format_text ,format_year,format_duration,)
from app.recommenders.popularity import (PopularityRecommender)
from app.recommenders.collaborative import (CollaborativeRecommender)
from app.recommenders.hybrid import HybridRecommender
from app.analytics import (RecommendationMetrics,RecommendationDiversity,RecommendationCharts,RecommendationInsights)
from app.xai.explainer import RecommendationExplainer
from app.services.catalog_service import CatalogService
from app.services.search_service import SearchService
from app.services.spotify_service import SpotifyService
from app.services.matching_service import MatchingService

#adding cache funtions
@st.cache_resource
def get_catalog_service(songs):
    return CatalogService(songs)

@st.cache_resource
def get_search_service(songs):
    return SearchService(songs)

def get_spotify_service():
    return SpotifyService()

def get_matching_service():
    return MatchingService()

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

@st.cache_resource
def get_hybrid_recommender(recommender,collaborative):

    return HybridRecommender(recommender,collaborative)

@st.cache_data
def load_spotify_dataset():

    loader = SpotifyDataLoader().load_data()

    processor = (
        SpotifyPreprocessor(loader.song_df)
        .clean_data()
        .prepare_audio_features()
    )

    return processor

def display_xai(selected_song, recommended_song, recommender_type):
    if recommender_type == "content":
        explanation = RecommendationExplainer.explain_content(selected_song,recommended_song)

    elif recommender_type == "hybrid":
        explanation = RecommendationExplainer.explain_hybrid(selected_song,recommended_song)

    elif recommender_type == "popularity":
        explanation = RecommendationExplainer.explain_popularity(recommended_song)

    else:
        explanation = RecommendationExplainer.explain_collaborative()

    with st.expander("💡 Why was this recommended?"):

        if "similarity_score" in explanation:
            # st.markdown("### 🎯 Similarity Score")
            st.metric("🎯 Similarity Score", f"{explanation['similarity_score']}%")

        if "matching_features" in explanation:
            st.markdown("### 🎵 Top Matching Features")
            for feature in explanation["matching_features"]:
                if isinstance(feature, dict):
                    st.write(
                        f"✅ {feature['feature']} "
                        f"({feature['similarity']}%)"
                    )
                else:
                    st.write(f"✅ {feature}")

        st.markdown("### 📝 Explanation")
        st.info(explanation["explanation"])



def display_recommendations(recommendations,songs,score_label="Recommendation Score",selected_song=None,recommender_type=None):

    if recommendations is None or recommendations.empty:
        st.warning("No recommendations found.")
        return
    # this function displays the recommendations in a structured format using Streamlit containers and columns. 
    # It iterates through each recommendation and presents the song details along with the score in a user-friendly layout.

    for _, row in recommendations.iterrows():
        with st.container(border=True):
            left, right = st.columns([3,2])

            with left:

                st.subheader(row["name"])
                st.write(f"Artist: {format_text(row.get('artists'))}")

                album = row.get("album_name")
                if pd.notna(album) and str(album).strip() != "":
                    st.write(f"Album: {format_text(album)}")
                else:
                    st.write("Album: Solo Track/ Single Release")

                source = row.get("source", "Unknown")
                source_icons = {
                    "Content": "🎯 Content-Based",
                    "Collaborative": "👥 Collaborative",
                    "Hybrid": "⭐ Hybrid",
                    "Popularity": "🔥 Popular"
                }

                st.caption(source_icons.get(source, source))

            with right:
                st.metric(label=score_label, value=f"{row['score']*100:.1f}%")
                st.write(f"Year: {format_year(row.get('year'))}")
                
                duration = row.get("duration_ms")
                if pd.notna(duration):
                    st.write(f"Duration: {format_duration(duration)}")
             
            if recommender_type is not None:
                if recommender_type != "collaborative":
                    match = songs.loc[songs["id"] == row["id"]]

                    if not match.empty:
                        recommended_song = match.iloc[0]

                        display_xai(selected_song,recommended_song,recommender_type)

def render_content_search(search,spotify,matching,):
    """
    Renders the search section for the Content-Based recommender.

    Returns
    -------
    selected_song : str | None
    recommend_local : bool
    """

    query = st.text_input("🔍 Search Song",placeholder="Type song, artist or album...")

    recommend_local = False
    selected_song = None

    if query:
        search_results = search.search(query)

        if search_results.empty:
            selected_song = spotify_fallback(query=query,
                                        search=search,
                                        spotify=spotify,
                                        matching=matching,
                                    )

        else:
            options = search.build_display_names(search_results)
            selected_song = st.selectbox("Matching Songs", options)

            if selected_song is not None:
                recommend_local = st.button("🎯 Recommend Songs",key="local_recommend")

    return selected_song, recommend_local

def generate_content_recommendations(selected_song,recommended_local,recommender,catalog,songs,top_n):
    """
    Generate recommendations for the Content-Based recommender.

    Handles both:
    1. Local song selection
    2. Spotify fallback matched song
    """

    matched_song = st.session_state.get("matched_song")

    # Decide which song to recommend from
    if matched_song is not None:
        song_name = matched_song["name"]

    elif selected_song is not None:
        song_name = catalog.get_song_name(selected_song)

    else:
        return

    # Only generate when requested

    if not (recommend_local or matched_song is not None):
        return

    recommendations = recommender.recommend(song_name=song_name,n=top_n)
    recommendations = catalog.enrich_recommendations(recommendations)

    st.session_state["last_recommendations"] = recommendations

    # Selected song for XAI

    if matched_song is not None:
        selected_song_row = matched_song
    else:
        selected_song_row = catalog.get_song_row(selected_song)

    display_recommendations(recommendations,
                            songs,
                            "Content Score",
                            selected_song=selected_song_row,
                            recommender_type="content"
                        )

def spotify_fallback(query, search, spotify, matching):
    """
    Handles Spotify fallback when a song is not found locally.

    Returns
    -------
    str | None
        Name of the matched local song if found,
        otherwise None.
    """

    st.warning("Song not found in local catalog.")

    spotify_track = spotify.search_song(query)

    if spotify_track is None:
        st.error("Song not found on Spotify.")
        return None

    st.success("Song found on Spotify!")

    # Spotify Metadata
    col1, col2,col3 = st.columns([4,3,3])

    with col1:
        st.markdown(f"#### {spotify_track['name']}")
        
        # if spotify_track.get("album_image"):
        #     st.image(
        #         spotify_track["album_image"],
        #         use_column_width=True)

    with col2:
        st.markdown(f"**Artist:** {spotify_track['artist']}")
        st.write(f"**Album:** {spotify_track['album']}")

    with col3:
        st.write(f"**Duration:** {format_duration(spotify_track['duration_ms'])}")
        st.write(f"**Release Date:** {spotify_track['release_date']}")

    # Wait for user action
    # if not st.button("🎵 Find Similar Songs",key="spotify_match"):
    #     return None

    btn1, btn2 = st.columns([5,1])

    with btn1:
        find_match = st.button("🎵 Find Similar Songs",key="spotify_match")

    with btn2:
        recommend = st.button("🎯 Recommend Songs",key="spotify_recommend")

    if find_match:
        # Stage 1
        candidates = search.get_candidates(spotify_track["name"],
                                    spotify_track["artist"],)

        # Debug
        # with st.expander("Candidate Songs"):
        #     st.dataframe(
        #         candidates[
        #             [
        #                 "name",
        #                 "artists",
        #                 "year",
        #             ]
        #         ]
        #     )

        # Stage 2

        matches = matching.find_best_match(spotify_track,candidates)

        if matches.empty:
            st.error("No confident match found in local recommendation catalog.")
            return None

        st.success(f"Matched with: {matches.iloc[0]['name']}")

        with st.expander("Matched Songs"):
            st.dataframe(matches[["name","artists","matching_score"]])

        st.session_state["matched_song"] = matches.iloc[0]

    # Return the matched LOCAL song
    # return matches.iloc[0]["name"]

    if recommend:
        matched_song = st.session_state.get("matched_song")
        if matched_song is None:
            st.warning("Please click 'Find Similar Songs' first.")
            return None

        return matched_song["name"]



   
# Page Configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide")

# Store the latest recommendations
if "last_recommendations" not in st.session_state:
    st.session_state["last_recommendations"] = None

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.success("Hybrid Music Recommendation System")

# Title
st.title("Music Recommendation System with GenAI & XAI")
st.write(
    """
        A Hybrid Music Recommendation System combining
        Content-Based Filtering,
        Collaborative Filtering,
        Popularity-Based Recommendation,
        Explainable AI (XAI)
        and Generative AI.
        """
)

# Load Dataset
processor = load_spotify_dataset()

summary = processor.dataset_summary()

songs = processor.get_dataframe()

catalog = get_catalog_service(songs)

search = get_search_service(songs)

spotify = get_spotify_service()

matching = get_matching_service() 

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


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Content",
        "🔥 Popular",
        "👥 Collaborative",
        "⭐ Hybrid",
        "📊 Recommendation Insights"
    ]
)


with tab1:
    st.subheader("Content-Based Recommendation")
    col1, col2 = st.columns([3,1])

    with col1:  
        selected_song,recommend_local =render_content_search(search=search,
                                                            spotify=spotify,
                                                            matching=matching,
                                                            )

    with col2:
        top_n = st.number_input("Top",min_value=5,max_value=20,value=10)

    generate_content_recommendations(selected_song=selected_song,
                                     recommended_local=recommend_local,
                                    recommender=recommender,
                                    catalog=catalog,
                                    songs=songs,
                                    top_n=top_n,
                                    )

    st.markdown("---")



with tab2:

    st.subheader("🔥 Most Popular Songs")

    top_n = st.slider("Top Songs",5,20,10,key="popular_slider")
    popular = popularity.recommend(top_n)
    popular = catalog.enrich_recommendations(popular)
    display_recommendations(popular,
                            songs,
                            "Popularity Score",
                            recommender_type="popularity"
                            )


with tab3:

    st.subheader("👥 Collaborative Recommendation")

    song = st.selectbox("Choose a Song",collaborative.available_songs(),key="collab_song")
    top_n = st.slider("Number of Recommendations",5,20,10,key="collab_slider")

    if st.button("Recommend",key="collab_button"):
        recommendations = collaborative.recommend(song,top_n)
        recommendations = catalog.enrich_recommendations(recommendations)
        st.session_state["last_recommendations"] = recommendations

        if recommendations is None:
            st.error("Song not found.")

        else:
            display_recommendations(recommendations,
                                    songs,
                                    "Collaborative Score",
                                    recommender_type="collaborative"
                                    )

with tab4:

    st.subheader("⭐ Hybrid Recommendation")

    song = st.selectbox("Choose Song",catalog.available_songs(),key="hybrid_song")

    top_n = st.slider("Number of Recommendations",5,20,10,key="hybrid_slider")
    content_weight = st.slider("Content Weight",0.0,1.0,0.6,0.1)

    collaborative_weight = 1 - content_weight
    hybrid.set_weights(content_weight,collaborative_weight)

    st.write(f"Content : {content_weight:.1f}")
    st.write(f"Collaborative : {collaborative_weight:.1f}")

    if st.button("Generate Hybrid Recommendations",key="hybrid_button"):
        
        recommendations = hybrid.recommend(song_name=catalog.get_song_name(song),top_n=top_n)
        recommendations = catalog.enrich_recommendations(recommendations)
         
        st.session_state["last_recommendations"] = recommendations
        
        if recommendations is None:
            st.error("No recommendations found.")

        else:
            display_recommendations(recommendations,
                                    songs,
                                    "Hybrid Score",
                                    selected_song=catalog.get_song_row(song),
                                    recommender_type="hybrid"
                                    )

with tab5:

    st.header("📊 Recommendation Insights")
    st.info("Generate recommendations from any recommender to see analytics.")

    recommendations = st.session_state.get("last_recommendations")

    if recommendations is None or recommendations.empty:
        st.warning("Generate recommendations first.")

    else:
        summary = RecommendationMetrics.recommendation_summary(recommendations)

        col1, col2 = st.columns(2)
        col1.metric("🎵 Songs",summary["total_songs"])
        col2.metric("👨‍🎤 Artists",summary["unique_artists"])

        col1, col2 = st.columns(2)
        col1.metric("⭐ Avg Score",f"{summary['average_score']:.1f}%")
        popularity = summary["average_popularity"]
        
        if popularity is not None:
            col2.metric("🔥 Avg Popularity",popularity)
        else:
            col2.metric("🔥 Avg Popularity","-")
            col2.caption("Popularity information is unavailable for these recommendations.")

        col1, col2 = st.columns(2)
        col1.metric("📅 Avg Year",summary["average_year"])
        col2.metric("🌍 Unique Years",summary["unique_years"])
        

        st.markdown("---")
        st.subheader("🌍 Recommendation Quality Analysis")

        diversity = RecommendationDiversity.diversity_summary(recommendations)

        col1, col2 = st.columns(2)

        with col1:

            diversity_card(
                "Overall Diversity",
                diversity["overall_diversity"],
                RecommendationInsights.overall(diversity["overall_diversity"]),
                "🌍"
            )

        with col2:

            diversity_card(
                "Artist Diversity",
                diversity["artist_diversity"],
                RecommendationInsights.artist(diversity["artist_diversity"]),
                "🎤"
            )

        col1, col2 = st.columns(2)

        with col1:
            diversity_card(
                "Year Diversity",
                diversity["year_diversity"],
                RecommendationInsights.year(diversity["year_diversity"]),
                "📅"
            )

        with col2:
            popularity_available = ( "popularity" in recommendations.columns 
                                    and recommendations["popularity"].notna().any())
        
            if popularity_available:
                diversity_card("Popularity Diversity",
                    diversity["popularity_diversity"],
                    RecommendationInsights.popularity(diversity["popularity_diversity"]),
                    "⭐"
                )

            else:

                diversity_card("Popularity Diversity",
                    0,
                    "Popularity information is unavailable for these recommendations.",
                    "⭐"
                )

    st.markdown("---")
    st.subheader("📈 Recommendation Analytics")
    if recommendations is not None and not recommendations.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.altair_chart(
                RecommendationCharts.artist_distribution(recommendations),
                use_container_width=True
            )

        with col2:
            st.altair_chart(
                RecommendationCharts.year_distribution(recommendations),
                use_container_width=True
            )

        col1, col2 = st.columns(2)

        with col1:
            chart = RecommendationCharts.popularity_distribution(recommendations)

            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Popularity information is unavailable for these recommendations.")


        with col2:
            st.altair_chart(
                RecommendationCharts.recommendation_scores(recommendations),
                use_container_width=True
            )   


    # st.subheader("📈 Recommendation Analytics")

    # st.write("Chart 1")
    # st.altair_chart(
    #     RecommendationCharts.artist_distribution(recommendations),
    #     use_container_width=True
    # )

    # st.write("Chart 2")
    # st.altair_chart(
    #     RecommendationCharts.year_distribution(recommendations),
    #     use_container_width=True
    # )

    # st.write("Chart 3")
    # st.altair_chart(
    #     RecommendationCharts.popularity_distribution(recommendations),
    #     use_container_width=True
    # )

    # st.write("Chart 4")
    # st.altair_chart(
    #     RecommendationCharts.recommendation_scores(recommendations),
    #     use_container_width=True
    # )
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