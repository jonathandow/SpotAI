import os
from typing import List, Dict, Any, Optional
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import warnings

# Suppress urllib3 LibreSSL/OpenSSL warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings('ignore', category=NotOpenSSLWarning)
except ImportError:
    pass
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress huggingface_hub deprecation warnings
try:
    from huggingface_hub.file_download import _deprecation_warning
    warnings.filterwarnings('ignore', message='`resume_download` is deprecated')
except Exception:
    pass

class SpotifyClient:
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080/callback')
        self.scopes = [
            'user-library-read',
            'user-top-read',
            'user-read-recently-played',
            'playlist-modify-public',
            'playlist-modify-private',
            'playlist-read-private',
            'playlist-read-collaborative'
        ]
        self.sp = None

    def get_auth_url(self) -> str:
        """Get the Spotify authorization URL."""
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join(self.scopes)
        )
        return auth_manager.get_authorize_url()

    def get_token(self, code: str) -> Dict[str, Any]:
        """Get access token from authorization code."""
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join(self.scopes)
        )
        return auth_manager.get_access_token(code)

    def set_token(self, token_info: Dict[str, Any]):
        """Set the Spotify client with token info."""
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join(self.scopes)
        )
        auth_manager.get_cached_token = lambda: token_info
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def get_top_tracks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's top tracks."""
        return self.sp.current_user_top_tracks(limit=limit)['items']

    def get_recent_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's recently played tracks."""
        return self.sp.current_user_recently_played(limit=limit)['items']

    def get_user_playlists(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's playlists."""
        playlists = []
        results = self.sp.current_user_playlists(limit=limit)
        while results:
            playlists.extend(results['items'])
            if results['next']:
                results = self.sp.next(results)
            else:
                break
        return playlists

    def get_playlist_tracks(self, playlist_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get tracks from a specific playlist."""
        tracks = []
        results = self.sp.playlist_tracks(playlist_id, limit=limit)
        while results:
            tracks.extend([item['track'] for item in results['items'] if item['track']])
            if results['next']:
                results = self.sp.next(results)
            else:
                break
        return tracks[:limit]

    def is_track_recently_played(self, track_id: str, days: int = 30) -> bool:
        """Check if a track was played recently."""
        recent_tracks = self.get_recent_tracks(limit=50)
        recent_track_ids = {item['track']['id'] for item in recent_tracks}
        return track_id in recent_track_ids

    def get_track_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Get audio features for tracks."""
        if not track_ids:
            return []
        features = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            try:
                batch_features = self.sp.audio_features(batch)
                if batch_features is None:
                    print(f"Spotify API returned None for audio_features batch: {batch}")
                    continue
                features.extend([f for f in batch_features if f])
            except spotipy.exceptions.SpotifyException as e:
                print(f"SpotifyException fetching audio features for batch: {e}")
                if e.http_status in (403, 404):
                    print("Spotify API 403/404 error. This may be due to expired/invalid token or incorrect track IDs.")
                continue
            except Exception as e:
                print(f"Error fetching audio features for batch: {e}")
                continue
        return features

    def get_tracks_by_features(self, feature_targets: List[Dict[str, float]], limit: int = 20) -> List[Dict[str, Any]]:
        """Get tracks matching the target features."""
        recommendations = []
        for target in feature_targets:
            try:
                # Validate seed track IDs if present
                seed_tracks = target.get('seed_tracks', [])
                valid_seeds = self.filter_valid_track_ids(seed_tracks) if seed_tracks else []
                params = dict(
                    limit=limit // len(feature_targets),
                    target_acousticness=target.get('acousticness'),
                    target_danceability=target.get('danceability'),
                    target_energy=target.get('energy'),
                    target_instrumentalness=target.get('instrumentalness'),
                    target_valence=target.get('valence'),
                    min_popularity=20
                )
                if valid_seeds:
                    params['seed_tracks'] = valid_seeds[:5]
                result = self.sp.recommendations(**params)
                recommendations.extend(result['tracks'])
            except spotipy.exceptions.SpotifyException as e:
                print(f"SpotifyException in get_tracks_by_features: {e}")
                if e.http_status in (403, 404):
                    print("Spotify API 403/404 error. This may be due to expired/invalid token or bad parameters.")
                continue
            except Exception as e:
                print(f"Error in get_tracks_by_features: {e}")
                continue
        return recommendations[:limit]

    def filter_valid_track_ids(self, track_ids: List[str]) -> List[str]:
        """Return only valid/available Spotify track IDs."""
        if not track_ids:
            return []
        try:
            results = self.sp.tracks(track_ids)
            valid_ids = [t['id'] for t in results['tracks'] if t and t.get('id')]
            return valid_ids
        except Exception as e:
            print(f"Error validating track IDs: {e}")
            return []

    def create_playlist(self, name: str, tracks: List[Dict[str, Any]], description: str = '') -> Optional[Dict[str, Any]]:
        """Create a new playlist with the given tracks."""
        try:
            # Create playlist
            user_id = self.sp.current_user()['id']
            playlist = self.sp.user_playlist_create(user_id, name, description=description)
            
            # Add tracks
            if tracks:
                track_uris = []
                for track in tracks:
                    if isinstance(track, dict) and 'uri' in track:
                        track_uris.append(track['uri'])
                    elif isinstance(track, str):
                        if track.startswith('spotify:track:'):
                            track_uris.append(track)
                        else:
                            track_uris.append(f'spotify:track:{track}')
                
                if track_uris:
                    for i in range(0, len(track_uris), 100):
                        batch = track_uris[i:i + 100]
                        self.sp.playlist_add_items(playlist['id'], batch)
            
            return playlist
        except Exception as e:
            print(f"Error creating playlist: {str(e)}")
            return None
