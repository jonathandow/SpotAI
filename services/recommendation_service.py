import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .spotify_service import SpotifyService
from .llm_service import LLMService

class RecommendationService:
    def __init__(self, spotify_service: SpotifyService, llm_service: LLMService):
        self.spotify = spotify_service
        self.llm = llm_service
        self.scaler = StandardScaler()

    def _extract_audio_features(self, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """Extract and normalize audio features from tracks."""
        try:
            # Get track IDs and fetch their audio features
            track_ids = [track['id'] for track in tracks if 'id' in track]
            if not track_ids:
                logging.warning('No valid track IDs found')
                return np.zeros((1, 7))  # Return default array if no valid tracks

            audio_features = self.spotify.get_audio_features(track_ids)
            if not audio_features:
                logging.warning('No audio features returned')
                return np.zeros((1, 7))

            # Extract relevant features
            features = []
            for feature in audio_features:
                if feature:
                    features.append([
                        feature.get('danceability', 0),
                        feature.get('energy', 0),
                        feature.get('loudness', 0),
                        feature.get('speechiness', 0),
                        feature.get('acousticness', 0),
                        feature.get('instrumentalness', 0),
                        feature.get('valence', 0)
                    ])

            if not features:
                logging.warning('No valid features extracted')
                return np.zeros((1, 7))

            # Convert to numpy array and normalize
            features_array = np.array(features)
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)

            return self.scaler.fit_transform(features_array)

        except Exception as e:
            logging.error(f'Error in _extract_audio_features: {str(e)}')
            return np.zeros((1, 7))

    def _cluster_tracks(self, features: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Cluster tracks based on their audio features."""
        try:
            if len(features) < n_clusters:
                logging.warning(f'Not enough samples ({len(features)}) for {n_clusters} clusters')
                n_clusters = max(2, len(features))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(features)
        except Exception as e:
            logging.error(f'Error in _cluster_tracks: {str(e)}')
            return np.zeros(len(features), dtype=int)

    def generate_recommendations(self, seed_playlist_id: str = None) -> Dict[str, Any]:
        """Generate personalized track recommendations."""
        try:
            # 1. Get seed tracks from user's top tracks if no playlist is provided
            if seed_playlist_id:
                seed_tracks = self.spotify.get_playlist_tracks(seed_playlist_id)
            else:
                seed_tracks = self.spotify.get_top_tracks(limit=50)

            if not seed_tracks:
                logging.error('No seed tracks available')
                return {
                    'tracks': [],
                    'preferences': 'Could not analyze music preferences.',
                    'description': 'Could not generate recommendations.'
                }

            # 2. Analyze user preferences
            preferences = self.llm.analyze_music_preferences(seed_tracks[:20])  # Analyze top 20 tracks
            
            # 3. Get audio features and cluster tracks
            try:
                features = self._extract_audio_features(seed_tracks)
                clusters = self._cluster_tracks(features)
                
                # 4. Get seed tracks from different clusters
                unique_clusters = np.unique(clusters)
                seed_track_ids = []
                
                # First try to get tracks from different clusters
                for cluster in unique_clusters:
                    cluster_indices = np.where(clusters == cluster)[0]
                    if len(cluster_indices) > 0:
                        # Take the most popular track from each cluster
                        cluster_tracks = [seed_tracks[i] for i in cluster_indices]
                        sorted_tracks = sorted(cluster_tracks, key=lambda x: x.get('popularity', 0), reverse=True)
                        if sorted_tracks and 'id' in sorted_tracks[0]:
                            seed_track_ids.append(sorted_tracks[0]['id'])
                    
                    if len(seed_track_ids) >= 5:  # Spotify allows max 5 seed tracks
                        break
                        
                # If we don't have enough tracks from clustering, just use the most popular ones
                if len(seed_track_ids) < 5:
                    logging.info('Not enough tracks from clustering, using most popular tracks')
                    sorted_tracks = sorted(seed_tracks, key=lambda x: x.get('popularity', 0), reverse=True)
                    for track in sorted_tracks:
                        if 'id' in track and track['id'] not in seed_track_ids:
                            seed_track_ids.append(track['id'])
                            if len(seed_track_ids) >= 5:
                                break
                                
                if not seed_track_ids:
                    raise ValueError('No valid seed tracks found')
                    
            except Exception as e:
                logging.error(f'Error in track clustering: {str(e)}')
                # Fallback to using the most popular tracks
                sorted_tracks = sorted(seed_tracks, key=lambda x: x.get('popularity', 0), reverse=True)
                seed_track_ids = [track['id'] for track in sorted_tracks[:5] if 'id' in track]

            # 5. Get base recommendations from Spotify
            base_recommendations = self.spotify.get_recommendations(seed_track_ids)
            
            if not base_recommendations:
                logging.warning('No recommendations returned from Spotify API')
                # Use the seed tracks as fallback recommendations
                base_recommendations = [self.spotify.get_track_info(track_id) for track_id in seed_track_ids]
                if not base_recommendations:
                    logging.error('Could not get any valid tracks')
                    return {
                        'tracks': [],
                        'preferences': preferences,
                        'description': 'Could not generate recommendations at this time.'
                    }

            # 6. Enhance recommendations using LLM
            try:
                recent_context = [item['track'] for item in self.spotify.get_recent_tracks(limit=10)]
                enhanced_recommendations = self.llm.enhance_recommendations(
                    base_recommendations,
                    preferences,
                    recent_context
                )
            except Exception as e:
                logging.error(f'Error enhancing recommendations: {str(e)}')
                enhanced_recommendations = base_recommendations

            # 7. Generate playlist description
            try:
                description = self.llm.generate_playlist_description(
                    enhanced_recommendations,
                    preferences
                )
            except Exception as e:
                logging.error(f'Error generating playlist description: {str(e)}')
                description = f'A personalized playlist created by SpotAI on {datetime.now().strftime("%Y-%m-%d")}'

            return {
                'tracks': enhanced_recommendations,
                'preferences': preferences,
                'description': description
            }
            
        except Exception as e:
            logging.error(f'Error in generate_recommendations: {str(e)}')
            return {
                'tracks': [],
                'preferences': 'Error analyzing music preferences.',
                'description': 'An error occurred while generating recommendations.'
            }
