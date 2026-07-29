from __future__ import annotations

import logging
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from app.config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
)

logger = logging.getLogger(__name__)


class SpotifyClient:
    """
    Wrapper around the Spotify Web API.

    Responsibilities:
    - Authenticate with Spotify
    - Search tracks
    - Fetch complete track metadata
    - Return data in our application's format
    """

    def __init__(self) -> None:
        self.client = self._create_client()

    def _create_client(self) -> spotipy.Spotify:
        """Authenticate and create a Spotify client."""

        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise ValueError("Spotify credentials not found. "
                            "Please check your .env file."
                            )

        auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,client_secret=SPOTIFY_CLIENT_SECRET,)

        return spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=10,
            retries=3,
        )

    def search_track(self,track_name: str,artist: Optional[str] = None,) -> Optional[dict]:
        """Search for a track and return the first matching result with complete metadata."""

        query = f'track:"{track_name}"'

        if artist:
            query += f' artist:"{artist}"'

        logger.info("Searching Spotify: %s", query)

        try:
            results = self.client.search(
                q=query,
                type="track",
                limit=1,
            )

            items = results.get("tracks", {}).get("items", [])

            if not items:
                logger.warning("No track found.")
                return None

            track_id = items[0]["id"]

            return self.get_track(track_id)

        except Exception as e:
            logger.exception("Spotify search failed.")
            raise RuntimeError(f"Spotify search failed: {e}") from e

    def get_track(self, track_id: str) -> dict:
        """Fetch a complete track object using its Spotify ID."""

        try:
            track = self.client.track(track_id)

            return self._format_track(track)

        except Exception as e:
            logger.exception("Failed to fetch track.")
            raise RuntimeError(f"Failed to fetch track: {e}") from e

    def get_artist(self, artist_id: str) -> dict:
        """Fetch an artist object."""
        return self.client.artist(artist_id)

    def get_album(self, album_id: str) -> dict:
        """Fetch an album object."""
        return self.client.album(album_id)

    def _format_track(self, track: dict) -> dict:
        """Convert Spotify's nested Track object into our standardized dictionary."""

        album = track.get("album", {})
        artists = track.get("artists", [])

        first_artist = artists[0] if artists else {}

        images = album.get("images", [])

        album_image = images[0]["url"] if images else None

        return {
            "id": track.get("id"),
            "name": track.get("name"),
            "artist": first_artist.get("name"),
            "artist_id": first_artist.get("id"),
            "album": album.get("name"),
            "album_id": album.get("id"),
            "album_image": album_image,
            "spotify_url": track.get("external_urls", {}).get("spotify"),
            "preview_url": track.get("preview_url"),
            "popularity": track.get("popularity"),
            "release_date": album.get("release_date"),
            "duration_ms": track.get("duration_ms"),
            "explicit": track.get("explicit"),
        }