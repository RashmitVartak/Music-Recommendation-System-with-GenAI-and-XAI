from __future__ import annotations
from difflib import SequenceMatcher
import pandas as pd

class MatchingService:
    """
    Maps a Spotify song to the closest song in the local merged dataset.
    """

    def __init__(self):
        pass

    MIN_MATCH_SCORE = 0.75

    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(
            None,
            str(a).lower(),
            str(b).lower()
        ).ratio()

    @staticmethod
    def year_score(local_year, spotify_year):
        if pd.isna(local_year) or spotify_year is None:
            return 0

        try:
            local_year = int(local_year)
            spotify_year = int(spotify_year)
        except Exception:
            return 0

        diff = abs(local_year - spotify_year)

        if diff == 0: return 1

        if diff <= 1: return 0.8

        if diff <= 2: return 0.6

        if diff <= 5: return 0.3

        return 0

    def find_best_match(self,spotify_track: dict,candidates: pd.DataFrame,top_k: int = 5,) -> pd.DataFrame:

        spotify_title = spotify_track["name"]
        spotify_artist = spotify_track["artist"]
        spotify_release = spotify_track.get("release_date")
        spotify_year = None

        if spotify_release:
            spotify_year = int(spotify_release[:4])

        scores = []

        for idx, row in candidates.iterrows():
            title_score = self.similarity(spotify_title,row["name"])
            artist_score = self.similarity(spotify_artist,row["artists"])
            release_score = self.year_score(row["year"],spotify_year,)

            final_score = (
                0.50 * title_score +
                0.40 * artist_score +
                0.10 * release_score
            )

            scores.append({"row_index": idx, "matching_score": final_score})

        score_df = pd.DataFrame(scores)

        score_df = score_df.sort_values("matching_score",ascending=False,)
        best = candidates.loc[score_df["row_index"].head(top_k)].copy()
        

        best["matching_score"] = (
            score_df["matching_score"]
            .head(top_k)
            .values
        )

        confident_matches = best[best["matching_score"] >= self.MIN_MATCH_SCORE]

        if not confident_matches.empty:
            return confident_matches.reset_index(drop=True)

        # No confident matches
        # Return the top ranked songs anyway
        return best.reset_index(drop=True)