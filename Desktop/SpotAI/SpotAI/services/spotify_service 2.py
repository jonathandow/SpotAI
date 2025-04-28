import os
import logging
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Any

class SpotifyService:
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = "http://localhost:8080/callback"
        self.scope = "user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-modify-private user-read-private user-read-email playlist-read-private"
        
        logging.info(f'Spotify client ID length: {len(self.client_id) if self.client_id else 0}')
        logging.info(f'Spotify client secret length: {len(self.client_secret) if self.client_secret else 0}')
        
        if not self.client_id or not self.client_secret:
            logging.error('Missing Spotify credentials. Please check your .env file.')
            raise ValueError('Missing Spotify credentials. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file.')
        
        self.auth_manager = None
        self.sp = None

    def get_auth_manager(self):
        if not self.auth_manager:
            logging.info('Creating new SpotifyOAuth manager')
            logging.info(f'Using client_id: {self.client_id}')
            logging.info(f'Using redirect_uri: {self.redirect_uri}')
            
            # Clean and validate credentials
            client_id = self.client_id.strip()
            client_secret = self.client_secret.strip()
            redirect_uri = self.redirect_uri.strip()
            
            logging.info(f'Cleaned client_id length: {len(client_id)}')
            logging.info(f'Cleaned client_secret length: {len(client_secret)}')
            logging.info(f'Cleaned redirect_uri: {redirect_uri}')
            
            self.auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=self.scope,
                open_browser=False,
                show_dialog=True,  # Force display of Spotify auth dialog
                cache_path=None  # Disable cache file
            )
            logging.info('SpotifyOAuth manager created successfully')
        return self.auth_manager

    def get_spotify_client(self):
        if not self.sp:
            logging.info('Creating new Spotify client')
            auth_manager = self.get_auth_manager()
            
            # Check if token needs refresh
            if auth_manager.is_token_expired(auth_manager.get_cached_token()):
                logging.info('Token expired, refreshing...')
                auth_manager.refresh_access_token(auth_manager.get_cached_token()['refresh_token'])
            
            self.sp = Spotify(auth_manager=auth_manager)
            logging.info('Spotify client created successfully')
        return self.sp


    def get_user_info(self) -> Dict[str, Any]:
        """Get current user's profile information."""
        return self.sp.current_user()

    def get_audio_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Get audio features for a list of tracks."""
        try:
            if not track_ids:
                logging.warning('No track IDs provided for audio features')
                return []
                
            # Process in batches of 100 (Spotify API limit)
            features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i + 100]
                logging.info(f'Getting audio features for batch of {len(batch)} tracks')
                batch_features = self.sp.audio_features(batch)
                if batch_features:
                    features.extend([f for f in batch_features if f is not None])
            return features
        except Exception as e:
            logging.error(f'Error getting audio features: {str(e)}')
            raise

    def get_top_tracks(self, limit: int = 50, time_range: str = 'medium_term') -> List[Dict[str, Any]]:
        """Get user's top tracks."""
        return self.sp.current_user_top_tracks(limit=limit, time_range=time_range)['items']

    def get_recent_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's recently played tracks."""
        return self.sp.current_user_recently_played(limit=limit)['items']

    def get_saved_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's saved tracks."""
        return self.sp.current_user_saved_tracks(limit=limit)['items']

    def create_playlist(self, user_id: str, name: str, description: str = '') -> Dict[str, Any]:
        """Create a new playlist."""
        try:
            logging.info(f'Creating playlist {name} for user {user_id}')
            playlist = self.sp.user_playlist_create(user_id, name, public=True, description=description)
            logging.info(f'Successfully created playlist {playlist["id"]}')
            return playlist
        except Exception as e:
            logging.error(f'Error creating playlist: {str(e)}')
            raise

    def add_tracks_to_playlist(self, playlist_id: str, track_uris: List[str]) -> None:
        """Add tracks to a playlist."""
        try:
            if not track_uris:
                logging.warning('No tracks provided to add to playlist')
                return
                
            logging.info(f'Adding {len(track_uris)} tracks to playlist {playlist_id}')
            # Add tracks in batches of 100 (Spotify API limit)
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i:i + 100]
                self.sp.playlist_add_items(playlist_id, batch)
                logging.info(f'Added batch of {len(batch)} tracks to playlist {playlist_id}')
        except Exception as e:
            logging.error(f'Error adding tracks to playlist: {str(e)}')
            raise

    def get_track_info(self, track_id: str) -> Dict[str, Any]:
        """Get detailed information about a track."""
        return self.sp.track(track_id)

    def get_artist_info(self, artist_id: str) -> Dict[str, Any]:
        """Get detailed information about an artist."""
        return self.sp.artist(artist_id)

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
