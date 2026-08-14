"""Responsible for search local dataset"""
from turtle import st, title

import pandas as pd

class SearchService:
    """Handles searching songs from the local merged dataset."""

    def __init__(self, songs: pd.DataFrame):
        self.songs = songs.copy()

        # Precompute lowercase columns for faster searching
        self.songs["_name_lower"] = (
            self.songs["name"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        self.songs["_artist_lower"] = (
            self.songs["artists"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        self.songs["_album_lower"] = (
            self.songs["album_name"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

    def search(self, query: str, limit: int = 25) -> pd.DataFrame:

        if not query:
            return pd.DataFrame()

        query = query.lower().strip()
        df = self.songs.copy()
        df = df.assign(priority=999)
        #setting priority for search results based on relevance
        # Exact title
        exact = df["_name_lower"] == query
        df.loc[exact, "priority"] = 1

        # Starts with title
        starts = (df["_name_lower"].str.startswith(query) & (~exact))
        df.loc[starts, "priority"] = 2

        # Artist
        artist = (df["_artist_lower"].str.contains(query, na=False) & (df["priority"] == 999))
        df.loc[artist, "priority"] = 3
    
        # Contains title
        contains = (df["_name_lower"].str.contains(query, na=False) & (~exact) & (~starts))
        df.loc[contains, "priority"] = 4

        # Album
        album = (df["_album_lower"].str.contains(query, na=False) & (df["priority"] == 999))
        df.loc[album, "priority"] = 5
        results = df[df["priority"] != 999]

        if results.empty:
            return results

        return (
            results
            .sort_values(
                by=["priority","popularity","year"],
                ascending=[True,False,False]
            ).head(limit)
        )

    def build_display_names(self, df: pd.DataFrame):
        """Convert dataframe into dropdown labels."""
        options = []
        for _, row in df.iterrows():
            year = row["year"]
            if pd.isna(year): year = "Unknown"

            else: year = int(year)

            options.append(f"{row['name']} — {row['artists']} ({year})")

        return options

    def get_candidates(self, title: str, artist: str = "", limit: int = 200) -> pd.DataFrame:
        """
        Returns a candidate pool for fuzzy matching.

        This method is optimized for Spotify fallback.
        """

        title = str(title).lower().strip()
        artist = str(artist).lower().strip()

        title_tokens = [token
            for token in title.split()
                if len(token) >= 3
        ]

        artist_tokens = [token
            for token in artist.split()
                if len(token) >= 3
        ]

        mask = pd.Series(False, index=self.songs.index)

        for token in title_tokens:
            mask |= self.songs["_name_lower"].str.contains(token,na=False,regex=False,case=False)

        for token in artist_tokens:
            mask |= self.songs["_artist_lower"].str.contains(token,na=False,regex=False,case=False)

        candidates = self.songs[mask]

        if candidates.empty:
            return candidates

        return (
            candidates
            .sort_values(by=["popularity", "year"],ascending=False)
            .head(limit)
        )