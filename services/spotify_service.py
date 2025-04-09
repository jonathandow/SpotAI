import os
import logging
from typing import List, Dict, Any, Optional
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

class SpotifyService:
    def __init__(self):
        self.sp = None
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080/callback')
        self.scopes = [
            'user-library-read',
            'user-top-read',
            'user-read-recently-played',
            'playlist-modify-public',
            'playlist-modify-private',
            'playlist-read-private'
        ]

    def get_auth_manager(self) -> SpotifyOAuth:
        """Get the SpotifyOAuth manager."""
        return SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join(self.scopes)
        )

    def get_spotify_client(self) -> Spotify:
        """Get an authenticated Spotify client."""
        if not self.sp:
            auth_manager = self.get_auth_manager()
            if not auth_manager.validate_token(auth_manager.get_cached_token()):
                auth_manager.get_access_token()
            self.sp = Spotify(auth_manager=auth_manager)
        return self.sp

    def get_user_info(self) -> Dict[str, Any]:
        """Get the current user's Spotify profile."""
        return self.sp.current_user()

    def get_top_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the user's top tracks."""
        return self.sp.current_user_top_tracks(limit=limit)['items']

    def get_recent_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the user's recently played tracks."""
        return self.sp.current_user_recently_played(limit=limit)['items']

    def get_track_info(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific track."""
        try:
            return self.sp.track(track_id)
        except Exception as e:
            logging.error(f'Error getting track info for {track_id}: {str(e)}')
            return None

    def get_audio_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Get audio features for a list of tracks."""
        try:
            # Filter out any None or empty track IDs
            valid_track_ids = [tid for tid in track_ids if tid]
            if not valid_track_ids:
                return []

            # Process in batches of 100 (Spotify API limit)
            features = []
            for i in range(0, len(valid_track_ids), 100):
                batch = valid_track_ids[i:i + 100]
                try:
                    batch_features = self.sp.audio_features(batch)
                    if batch_features:
                        features.extend([f for f in batch_features if f])
                except Exception as e:
                    logging.warning(f'Error getting audio features for batch: {str(e)}')

            return features
        except Exception as e:
            logging.error(f'Error in get_audio_features: {str(e)}')
            return []

    def create_playlist(self, name: str, tracks: List[Dict[str, Any]], description: str = '') -> Optional[Dict[str, Any]]:
        """Create a new playlist with the given tracks."""
        try:
            user_id = self.sp.current_user()['id']
            playlist = self.sp.user_playlist_create(user_id, name, description=description)
            if tracks:
                track_uris = [track['uri'] for track in tracks if 'uri' in track]
                if track_uris:
                    # Add tracks in batches of 100 (Spotify API limit)
                    for i in range(0, len(track_uris), 100):
                        batch = track_uris[i:i + 100]
                        self.sp.playlist_add_items(playlist['id'], batch)
            return playlist
        except Exception as e:
            logging.error(f'Error creating playlist: {str(e)}')
            return None

    def get_recommendations(self, seed_tracks: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Get track recommendations based on seed tracks."""
        try:
            # Validate seed tracks exist
            valid_tracks = []
            for track_id in seed_tracks:
                try:
                    track = self.sp.track(track_id)
                    if track:
                        valid_tracks.append(track_id)
                except Exception as e:
                    logging.warning(f'Invalid track ID {track_id}: {str(e)}')
            
            if not valid_tracks:
                logging.error('No valid seed tracks found')
                # Get some popular tracks as fallback
                top_tracks = self.get_top_tracks(limit=5)
                valid_tracks = [track['id'] for track in top_tracks]
            
            # Use at most 5 seed tracks (Spotify API limit)
            valid_tracks = valid_tracks[:5]
            logging.info(f'Using seed tracks: {valid_tracks}')
            
            try:
                recommendations = self.sp.recommendations(seed_tracks=valid_tracks, limit=limit)
                return recommendations['tracks']
            except Exception as e:
                logging.error(f'Error getting recommendations: {str(e)}')
                # Return the seed tracks as fallback
                return [self.sp.track(track_id) for track_id in valid_tracks]
                
        except Exception as e:
            logging.error(f'Error in get_recommendations: {str(e)}')
            return []
