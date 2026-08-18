from typing import Optional

from app.spotify.spotify_client import SpotifyClient


class SpotifyService:
    """
    Business logic layer for Spotify.

    This service hides SpotifyClient from the UI.
    All Spotify-related operations should go through here.
    """

    def __init__(self):
        self.client = SpotifyClient()

    def search_song(self, query: str) -> Optional[dict]:
        """
        Search for a song on Spotify.

        Parameters
        ----------
        query : str
            Song title entered by the user.

        Returns
        -------
        dict | None
            Normalized track information returned by SpotifyClient.
        """

        if not query or not query.strip():
            return None

        try:
            return self.client.search_track(query.strip())

        except Exception as e:
            return None

    def song_exists(self, query: str) -> bool:
        """Check whether Spotify can find the song."""
        result = self.search_song(query)

        return result is not None

    def format_track(self, track: dict) -> dict:
        """Returns a UI-friendly dictionary."""

        if track is None:
            return {}

        return {
            "id": track.get("id"),
            "name": track.get("name"),
            "artist": track.get("artist"),
            "album": track.get("album"),
            "release_date": track.get("release_date"),
            "duration_ms": track.get("duration_ms"),
            "duration": self._format_duration(track.get("duration_ms")),
            "popularity": track.get("popularity"),
            "explicit": track.get("explicit"),
            "album_image": track.get("album_image"),
            "preview_url": track.get("preview_url"),
        }

    @staticmethod
    def _format_duration(duration_ms):

        if duration_ms is None:
            return "Unknown"

        total_seconds = int(duration_ms / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60

        return f"{minutes}:{seconds:02d}"