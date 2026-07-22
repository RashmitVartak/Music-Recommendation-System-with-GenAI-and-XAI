import altair as alt
import plotly.graph_objects as go

class RecommendationCharts:

    @staticmethod
    def artist_distribution(df):
        """
        Top recommended artists.
        """
        artist_counts = (df["artists"].value_counts().head(10).reset_index())
        artist_counts.columns = ["Artist", "Count"]

        chart = (
            alt.Chart(artist_counts)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Recommendations"),
                y=alt.Y("Artist:N", sort="-x", title="Artist"),
                tooltip=["Artist", "Count"]
            )
            .properties(title="🎤 Top Recommended Artists", height=350)
            )

        return chart

    @staticmethod
    def year_distribution(df):
        """
        Trend of recommendation years.
        """

        year_counts = (
            df["year"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        year_counts.columns = ["Year", "Count"]

        chart = (
            alt.Chart(year_counts).mark_line(point=True, interpolate="monotone").encode(
                                    x=alt.X("Year:O", title="Release Year"),
                                    y=alt.Y("Count:Q", title="Songs"),
                                    tooltip=["Year", "Count"]
                                ).properties(title="📅 Recommendation Year Trend",height=350)
                            )

        return chart

    @staticmethod
    def popularity_distribution(df):
        """
        Histogram of popularity scores.
        """
        if ("popularity" not in df.columns or df["popularity"].dropna().empty):
            return None
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("popularity:Q", bin=alt.Bin(maxbins=15), title="Popularity"),
                y=alt.Y("count():Q", title="Number of Songs",axis=alt.Axis(format="d",tickMinStep=1)),
                tooltip=[alt.Tooltip("count():Q",title="Songs")]
            ).properties(title="⭐ Popularity Distribution",height=350)
        )

        return chart
    
    @staticmethod
    def recommendation_scores(df):

        scores = (df.sort_values("score", ascending=False))
        chart = (
            alt.Chart(scores).mark_bar(cornerRadius=8)
            .encode(
                x=alt.X("score:Q",title="Recommendation Score"),
                y=alt.Y("name:N",sort="-x",title="Song"),
                tooltip=["name","artists",alt.Tooltip("score:Q", format=".3f")]
            )
            .properties(title="🏆 Recommendation Scores",height=350)
        )

        return chart

class RecommendationGauge:

    @staticmethod
    def gauge(title, value):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": "%"},
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"thickness": 0.20},
                    "steps": [
                        {"range": [0, 40],"color": "#ef4444"},
                        {"range": [40, 70],"color": "#facc15"},
                        {"range": [70, 100],"color": "#22c55e"}
                    ]
                }
            )
        )

        fig.update_layout(
            height=220,
            margin=dict(
                l=15,
                r=15,
                t=50,
                b=15
            )
        )

        return fig