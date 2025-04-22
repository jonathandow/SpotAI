import logging
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
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
            # Filter out None or invalid tracks
            valid_tracks = [track for track in tracks if track and 'id' in track]
            if not valid_tracks:
                logging.warning('No valid tracks provided for feature extraction')
                # Return a 2D array with default features
                return np.zeros((1, 9))  # 9 features
            
            track_ids = [track['id'] for track in valid_tracks]
            logging.info(f'Getting audio features for {len(track_ids)} tracks')
            
            audio_features = self.spotify.get_audio_features(track_ids)
            if not audio_features:
                logging.warning('No audio features returned from Spotify')
                return np.zeros((1, 9))  # 9 features
            
            # Extract relevant numerical features
            features = []
            numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            
            for feat in audio_features:
                if feat and all(key in feat for key in numerical_features):
                    features.append([feat[key] for key in numerical_features])
                
            if not features:
                logging.warning('No valid features extracted from audio features')
                return np.zeros((1, 9))  # 9 features
            
            # Convert to numpy array
            features_array = np.array(features)
            
            # Ensure we have a 2D array
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            
            # Normalize features
            return self.scaler.fit_transform(features_array)
            
        except Exception as e:
            logging.error(f'Error extracting audio features: {str(e)}')
            return np.zeros((1, 9))  # 9 features

    def _cluster_tracks(self, features: np.ndarray, n_clusters: int = 5) -> List[int]:
        """Cluster tracks based on their audio features."""
        try:
            if features.shape[0] < n_clusters:
                logging.warning(f'Not enough samples ({features.shape[0]}) for {n_clusters} clusters. Reducing number of clusters.')
                n_clusters = max(1, features.shape[0])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(features)
        except Exception as e:
            logging.error(f'Error clustering tracks: {str(e)}')
            return [0] * features.shape[0]  # Return all tracks in same cluster

    def generate_recommendations(self, source: str = 'top') -> Dict[str, Any]:
        """
        Generate personalized recommendations using both Spotify data and LLM insights.
        
        Args:
            source: Source of seed tracks ('top', 'recent', or 'saved')
        
        Returns:
            Dictionary containing playlist information and tracks
        """
        try:
            # 1. Gather user's tracks based on source
            if source == 'top':
                seed_tracks = self.spotify.get_top_tracks()
            elif source == 'recent':
                seed_tracks = [item['track'] for item in self.spotify.get_recent_tracks()]
            else:  # saved
                seed_tracks = [item['track'] for item in self.spotify.get_saved_tracks()]

            # 2. Analyze user preferences using LLM
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
            logging.error(f"Error generating recommendations: {str(e)}")
            raise
