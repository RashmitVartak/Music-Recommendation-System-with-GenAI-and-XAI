import pandas as pd
'''Responsible for
-metadata
-dropdown display
-enrichment
-lookup'''
class CatalogService:
    """
    Handles song catalog operations such as
    - dropdown display
    - duplicate handling
    - metadata lookup
    """
    #private methods
    def __init__(self, songs: pd.DataFrame):
        self.songs = songs.reset_index(drop=True)

        self.catalog = self.songs.copy()
        self.catalog["df_index"] = self.catalog.index
        self.catalog["display_name"] = self.catalog.apply(
            self._format_display_name,
            axis=1
        )

        # display string -> dataframe row
        self.lookup = (
            self.catalog
            .set_index("display_name")
            .to_dict("index")
        )

    # Display Formatting
    @staticmethod
    def _format_display_name(row):

        title = row.get("name", "Unknown")
        artist = row.get("artists", "Unknown")
        year = row.get("year")

        if pd.isna(year) or year in [0, "", "0"]:
            year = "Unknown"
        else:
            year = int(float(year))

        return f"{title} — {artist} ({year})"

    # Public Methods
    def available_songs(self):
        """Returns formatted song names for dropdown."""

        return sorted(self.catalog["display_name"].tolist())

    def format_songs(self, songs: pd.DataFrame):
        """Returns formatted display names for a filtered song DataFrame."""

        return [
            self._format_display_name(row)
            for _, row in songs.iterrows()
        ]

    def get_song(self, display_name):
        """Returns complete song information."""

        return self.lookup.get(display_name)

    def get_song_name(self, display_name):
        """Returns original song title."""
        song = self.lookup.get(display_name)
        if song is None:
            return None

        return song["name"]

    def get_song_id(self, display_name):
        song = self.lookup.get(display_name)
        if song is None:
            return None

        return song["id"]

    def get_song_index(self, display_name):
        """Returns dataframe index of selected song."""
        song = self.lookup.get(display_name)

        if song is None:
            return None

        return song["df_index"]

    def get_song_row(self, display_name):
        song = self.lookup.get(display_name)

        if song is None:
            return None

        return self.songs.iloc[song["df_index"]]

    def enrich_recommendations(self, recommendations: pd.DataFrame):
        """Enrich recommendation results with metadata from merged_dataset."""
        if recommendations is None or recommendations.empty:
            return recommendations

        metadata_columns = [
            "id",
            "album_name",
            "genres",
            "duration_ms",
            "explicit",
            "release_date",
            "artist_popularity",
            "artist_followers",
        ]

        metadata = (
            self.songs[metadata_columns]
            .drop_duplicates(subset="id")
        )

        recommendations = recommendations.merge(
            metadata,
            on="id",
            how="left"
        )

        return recommendations