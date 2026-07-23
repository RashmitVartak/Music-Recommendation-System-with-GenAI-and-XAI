import streamlit as st
import pandas as pd

from app.data_loader import SpotifyDataLoader
from app.preprocessing import SpotifyPreprocessor
from app.recommenders.content_based import ContentBasedRecommender
from app.utils import format_number , diversity_card
from app.recommenders.popularity import (PopularityRecommender)
from app.recommenders.collaborative import (CollaborativeRecommender)
from app.recommenders.hybrid import HybridRecommender
from app.analytics import (RecommendationMetrics,RecommendationDiversity,RecommendationCharts,RecommendationInsights)
from app.xai.explainer import RecommendationExplainer

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

@st.cache_resource
def get_hybrid_recommender(
    recommender,
    collaborative
):
    return HybridRecommender(
        recommender,
        collaborative
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
    # this function displays the recommendations in a structured format using Streamlit containers and columns. It iterates through each recommendation and presents the song details along with the score in a user-friendly layout.
    for _, row in recommendations.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.subheader(row["name"])
                st.write(f"**Artist:** {row['artists']}")
                st.write(f"**Year:** {row['year']}")

                if pd.notna(row.get("popularity")):
                    st.write(f"**Popularity:** {row['popularity']}")

                source = row.get("source", "Unknown")
                st.caption(f"Source: {source}")

            with c2:
                st.metric(score_label,f"{row['score']*100:.1f}%")

            if recommender_type is not None:
                # Do not show XAI for Collaborative recommendations
                if recommender_type == "collaborative":
                    pass
                else:
                    match = songs.loc[songs["id"] == row["id"]]

                    if not match.empty:
                        recommended_song = match.iloc[0]
                        display_xai(selected_song,recommended_song,recommender_type)


# Page Configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

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
    st.markdown("Find Similar Songs")

    col1, col2 = st.columns([3,1])

    with col1:
        song = st.selectbox("Choose a Song",recommender.available_songs())

    with col2:
        top_n = st.number_input("Top",min_value=5,max_value=20,value=10)

    if st.button("Recommend Songs"):
        recommendations = recommender.recommend(song_name=song,n=top_n)
        st.session_state["last_recommendations"] = recommendations
        display_recommendations(recommendations,
                                songs,
                                "Content Score",
                                selected_song=songs.loc[songs["name"] == song].iloc[0],
                                recommender_type="content"
                                )

    st.markdown("---")



with tab2:

    st.subheader("🔥 Most Popular Songs")

    top_n = st.slider("Top Songs",5,20,10,key="popular_slider")
    popular = popularity.recommend(top_n)
    
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

    song = st.selectbox("Choose Song",recommender.available_songs(),key="hybrid_song")

    top_n = st.slider("Number of Recommendations",5,20,10,key="hybrid_slider")
    content_weight = st.slider("Content Weight",0.0,1.0,0.6,0.1)

    collaborative_weight = 1 - content_weight
    hybrid.set_weights(content_weight,collaborative_weight)

    st.write(f"Content : {content_weight:.1f}")
    st.write(f"Collaborative : {collaborative_weight:.1f}")

    if st.button("Generate Hybrid Recommendations",key="hybrid_button"):
        
        recommendations = hybrid.recommend(song,top_n)
        
        st.session_state["last_recommendations"] = recommendations

        if recommendations is None:
            st.error("No recommendations found.")

        else:
            display_recommendations(recommendations,
                                    songs,
                                    "Hybrid Score",
                                    selected_song=songs.loc[songs["name"] == song].iloc[0],
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