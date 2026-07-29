from pprint import pprint

from app.spotify import SpotifyClient


def main():
    spotify = SpotifyClient()

    # song = spotify.search_track("Believer")
    track = spotify.client.track("0pqnGHJpmpxLKifKRmU6WP")

    pprint(track.keys())


if __name__ == "__main__":
    main()