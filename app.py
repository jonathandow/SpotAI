import os
import random
from collections import Counter
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
from flask import Flask, redirect, request, session, url_for, render_template, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask_caching import Cache
import concurrent.futures
import time
import logging
from logging.handlers import RotatingFileHandler
from flask_pymongo import PyMongo
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient

handler = RotatingFileHandler('spotai.log', maxBytes=10000000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
handler.setFormatter(formatter)


#OPTIMAL CLUSTERS: 4-5
#FIGURING OUT OPTIMAL ITERATIONS
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_COOKIE_NAME'] = 'SpotAI'
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
file_path = "./info.txt"

with open(file_path, 'r') as f:
    CLIENT_ID = f.readline().strip()
    CLIENT_SECRET = f.readline().strip()
    app.config["MONGO_URI"] = f.readline().strip()

client = MongoClient(app.config["MONGO_URI"], 27017)
db = client.SpotAI

audio_features = db.audio_features
top_tracks = db.top_tracks
recent_tracks = db.recent_tracks
saved_tracks = db.saved_tracks

def ensure_mongo_connection():
    try:
        client.admin.command('ping')
        app.logger.info("MongoDB Connection established.")
    except ConnectionFailure:
        app.logger.error("MongoDB Connection failed.")
        raise

@app.before_request
def before_request():
    ensure_mongo_connection()

REDIRECT_URI = "http://localhost:5000/callback"

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3600})

class SpotAI:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                            client_secret=client_secret,
                                                            redirect_uri=redirect_uri,
                                                            scope="user-library-read user-read-recently-played user-top-read playlist-modify-public"))
    def clear_log_file(self, log_file):
        with open(log_file, 'w'):
            pass

    def get_token(self):
        token = session.get('token', None)
        if not token:
            return None
        if self.sp.auth_manager.is_token_expired(token):
            token = self.sp.auth_manager.refresh_token(token['refresh_token'])
        return token

    def save_user_data(self, collection_name, data_type, data):
        db[collection_name].update_one(
            {'_id': data_type},
            {'$set': {data_type: data}},
            upsert=True
        )

    def get_user_data(self, collection_name, data_type):
        try:
            collection_names = db.list_collection_names()
            if collection_name not in collection_names:
                db.create_collection(collection_name)
                app.logger.info(f"Created collection '{collection_name}' in MongoDB.")
            user_collection = db[collection_name]
            user_data = user_collection.find_one({'_id': data_type})

            return user_data[data_type] if user_data else None

        except ConnectionFailure as e:
            raise ConnectionError(f"MongoDB connection failed: {e}")

        except Exception as ex:
            raise ValueError(f"Error retrieving user data: {ex}")

    def get_audio_features_for_tracks(self, user_id, track_ids):
        app.logger.info("Getting audio features")
        stored_features = self.get_user_data('audio_features', user_id)
        if stored_features:
            stored_track_ids = {feature['id'] for feature in stored_features}
            track_ids_to_fetch = [track_id for track_id in track_ids if track_id not in stored_track_ids]
        else:
            stored_features = []
            track_ids_to_fetch = track_ids
        
        features = stored_features
        if track_ids_to_fetch:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.sp.audio_features, track_ids[i:i + 100]) for i in range(0, len(track_ids), 100)]
                for future in concurrent.futures.as_completed(futures):
                    features.extend(future.result())
            self.save_user_data('audio_features', user_id, features)
        
        return features
    
    def sync_user_data(self, user_id, collection_name, fetch_function):
        current_data = fetch_function()
        stored_data = self.get_user_data(collection_name, user_id)

        if not stored_data or [track for track in current_data] != [track for track in stored_data]:
            self.save_user_data(collection_name, user_id, current_data)
            return current_data
        return stored_data
    
    def get_all_top_tracks(self, user_id):
        return self.sync_user_data(user_id, 'top_tracks', self.fetch_all_top_tracks)

    def fetch_all_top_tracks(self):
        app.logger.info("Fetching top tracks")
        top = []
        results = self.sp.current_user_top_tracks(limit=50, time_range='medium_term')
        top.extend(results['items'])
        while results['next']:
            results = self.sp.next(results)
            top.extend(results['items'])
        return top
    
    def get_playlist_tracks(self, user_id, playlist_id):
        return self.sync_user_data(user_id, f'playlist_{playlist_id}_tracks', lambda: self.fetch_playlist_tracks(playlist_id))
    
    def fetch_playlist_tracks(self, playlist_id):
        app.logger.info(f"Fetching playlist tracks for playlist ID: {playlist_id}")
        tracks = []
        results = self.sp.playlist_tracks(playlist_id, limit=100)
        tracks.extend(results['items'])
        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])
        return tracks
    
    def get_recent_tracks(self, user_id):
        return self.sync_user_data(user_id, 'recent_tracks', self.fetch_recent_tracks)

    def fetch_recent_tracks(self):
        app.logger.info("Fetching recent tracks")
        recents = []
        results = self.sp.current_user_recently_played(limit=50)
        recents.extend(results['items'])
        while results['next']:
            results = self.sp.next(results)
            recents.extend(results['items'])
        return recents
    
    def get_all_saved_tracks(self, user_id):
        return self.sync_user_data(user_id, 'saved_tracks', self.fetch_all_saved_tracks)
    
    def fetch_all_saved_tracks(self):
        app.logger.info("Fetching saved tracks")
        saved = []
        results = self.sp.current_user_saved_tracks(limit=50)
        saved.extend(results['items'])
        while results['next']:
            results = self.sp.next(results)
            saved.extend(results['items'])
        return saved

    def create_playlist(self, user_id, playlist_name, num_clusters=5, num_iterations=10):
        self.clear_log_file('spotai.log')
        print("Creating Playlist......")
        start_time = time.time()
        top_artists = self.sp.current_user_top_artists(limit=5, time_range='medium_term')['items']

        #getting all tracks form user's library (thinking of not using recent_songs not sure)
        liked_songs = self.get_all_saved_tracks(user_id)
        recent_songs = self.get_recent_tracks(user_id)
        top_songs = self.get_all_top_tracks(user_id)


        #appending all of these tracks to check if the user already knows them
        known_names = {item['track']['name'] for item in liked_songs}
        known_names.update({item['track']['name'] for item in recent_songs})
        known_names.update({item['name'] for item in top_songs})

        known_tracks = {item['track']['id'] for item in liked_songs}
        known_tracks.update({item['track']['id'] for item in recent_songs})
        known_tracks.update({item['id'] for item in top_songs})

        track_ids = [track['id'] for track in top_songs]
        #getting audio features for the tracks
        try:
            features = self.get_audio_features_for_tracks(user_id, track_ids)
        except spotipy.exceptions.SpotifyException as e:
            return jsonify({'error': str(e)}), 500
        
         # Filter out None values from features
        features = [feature for feature in features if feature is not None]

        if not features:
            return jsonify({'error': "No audio features retrieved for the tracks"}), 500

        features_df = pd.DataFrame(features)
        # X = features_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

        plays = Counter(track_ids)
        #normalizing the play count
        max_plays = max(plays.values())
        features_df['play count'] = features_df['id'].apply(lambda x: plays[x] / max_plays)
        #selecting the features to be used for clustering
        feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'play count']
        X = features_df[feature_columns]
        print("Scaler...")
        #scaling the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("PCA...")
        #applying PCA to reduce the dimensions
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)
        print("KMeans...")
        #applying KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X_pca)
        features_df['kmeans_cluster'] = kmeans.labels_
        centroids_kmeans = kmeans.cluster_centers_
        print("GMM...")
        #applying GMM clustering
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        gmm.fit(X_pca)
        features_df['cluster'] = gmm.predict(X_pca)
        centroids = gmm.means_


        #combining the centroids of KMeans and GMM
        kmeans_weights = np.bincount(features_df['kmeans_cluster'])
        gmm_weights = np.bincount(features_df['cluster'])

        weighted_kmeans_centroids = np.dot(kmeans_weights, centroids_kmeans) / np.sum(kmeans_weights)
        weighted_gmm_centroids = np.dot(gmm_weights, centroids) / np.sum(gmm_weights)

        combined_centroids = (weighted_kmeans_centroids + weighted_gmm_centroids) / 2

        print("Recommendations...")
        #getting the recommendations
        recommendations = []
        for i in range(num_clusters):
            centroid = combined_centroids[i]
            cluster_tracks = features_df[features_df['cluster'] == i]
            
            if len(cluster_tracks) > 0:
                cluster_indices = cluster_tracks.index.tolist()
                cluster_distances = np.sum((X_pca[cluster_indices] - centroid) ** 2, axis=1)
                closest_index = cluster_indices[np.argmin(cluster_distances)]
                
                if closest_index < len(track_ids):
                    closest_id = track_ids[closest_index]
                    recommendations.append(closest_id)
                else:
                    app.logger.warning(f"Closest index {closest_index} out of range for track_ids. Skipping recommendation.")
            else:
                app.logger.warning(f"No tracks found in cluster {i}. Skipping recommendation.")

        user_id = self.sp.me()['id']
        playlist_description = f"SpotAI Recommendations. Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        playlists = self.sp.user_playlists(user_id)['items']
        playlist_id = None
        for playlist in playlists:
            if playlist['name'] == playlist_name:
                playlist_id = playlist['id']
                self.sp.playlist_replace_items(playlist_id, [])
                self.sp.playlist_change_details(playlist_id, description=playlist_description)
                break
        if not playlist_id:
            new_playlist = self.sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)
            playlist_id = new_playlist['id']

        rec_counter = Counter()
        for x in range(num_iterations):
            random_tracks = random.sample(recommendations, min(5, len(recommendations)))
            try:
                recommended_tracks = self.sp.recommendations(seed_tracks=random_tracks, limit=100)
            except spotipy.exceptions.SpotifyException as e:
                return jsonify({'error': str(e)}), 500

            new_recommendations = [track['id'] for track in recommended_tracks['tracks'] if track['id'] not in known_tracks and track['name'] not in known_names]
            rec_counter.update(new_recommendations)

        most_common_recommendations = [track_id for track_id, count in rec_counter.most_common(100)]
        self.sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist_id, tracks=most_common_recommendations)

        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        print(f"Playlist generation took {duration:.2f} seconds")  # Print the duration

        return render_template('playlist.html', playlist_name=playlist_name, playlist_url=f"https://open.spotify.com/playlist/{playlist_id}")

    def create_playlist_from_playlist(self, user_id, selected_playlist_id, playlist_name, num_clusters=5, num_iterations=10):
        start_time = time.time()
        self.clear_log_file('spotai.log')
        app.logger.info(f"Creating playlist from playlist_id: {selected_playlist_id}")
        cache_key = f'playlist_tracks_{selected_playlist_id}'
        cache.delete(cache_key)
        tracks = self.get_playlist_tracks(user_id, selected_playlist_id)
        # print(selected_playlist_id)
        for item in tracks:
            if item and 'track' in item and item['track']:
                track_name = item['track'].get('name')
                app.logger.info(f"Track name: {track_name}")
            else:
                app.logger.info("Invalid track: %s", item)

        track_ids = []
        for track in tracks:
            if track is not None and 'track' in track and track['track'] is not None:
                track_ids.append(track['track']['id'])
            else:
                app.logger.info(f"Skipping Invalid track: {track}")

        if not track_ids:
            return jsonify({'error': "No valid tracks found in the playlist"}), 500

        liked_songs = self.get_all_saved_tracks(user_id)
        recent_songs = self.get_recent_tracks(user_id)
        top_songs = self.get_all_top_tracks(user_id)

        known_names = {item['track']['name'] for item in liked_songs}
        known_names.update({item['track']['name'] for item in recent_songs[:100]})
        known_names.update({item['name'] for item in top_songs})

        known_tracks = {item['track']['id'] for item in liked_songs}
        known_tracks.update({item['track']['id'] for item in recent_songs[:100]})
        known_tracks.update({item['id'] for item in top_songs})

        # track_ids = [track['track']['id'] for track in tracks]

        try:
            features = self.get_audio_features_for_tracks(user_id, track_ids)
        except spotipy.exceptions.SpotifyException as e:
            return jsonify({'error': str(e)}), 500
        
        if not features:
            return jsonify({'error': "No audio features retrieved for the tracks"}), 500

        features_df = pd.DataFrame(features)
        # X = features_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

        plays = Counter(track_ids)

        max_plays = max(plays.values())
        features_df['play count'] = features_df['id'].apply(lambda x: plays[x] / max_plays)
        
        feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'play count']
        X = features_df[feature_columns]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X_pca)
        features_df['kmeans_cluster'] = kmeans.labels_
        centroids_kmeans = kmeans.cluster_centers_

        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        gmm.fit(X_pca)
        features_df['cluster'] = gmm.predict(X_pca)
        centroids = gmm.means_

        kmeans_weights = np.bincount(features_df['kmeans_cluster'])
        gmm_weights = np.bincount(features_df['cluster'])

        weighted_kmeans_centroids = np.dot(kmeans_weights, centroids_kmeans) / np.sum(kmeans_weights)
        weighted_gmm_centroids = np.dot(gmm_weights, centroids) / np.sum(gmm_weights)

        combined_centroids = (weighted_kmeans_centroids + weighted_gmm_centroids) / 2

        # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # kmeans.fit(X)
        # features_df['cluster'] = kmeans.labels_
        # centroids = kmeans.cluster_centers_

        recommendations = []
        # for i in range(num_clusters):
        #     centroid = combined_centroids[i]
        #     cluster_tracks = features_df[features_df['cluster'] == i]
        #     cluster_tracks_pca = X_pca[features_df['cluster'] == i]
        #     centroid_array = np.array(centroid)
        #     closest_index = np.argmin(np.sum((cluster_tracks_pca - centroid_array) ** 2, axis=1))
        #     closest_id = track_ids[cluster_tracks.index[closest_index]]
        #     recommendations.append(closest_id)
        for i in range(num_clusters):
            centroid = combined_centroids[i]
            cluster_tracks = features_df[features_df['cluster'] == i]
            
            if len(cluster_tracks) > 0:
                cluster_indices = cluster_tracks.index.tolist()
                cluster_distances = np.sum((X_pca[cluster_indices] - centroid) ** 2, axis=1)
                closest_index = cluster_indices[np.argmin(cluster_distances)]
                
                if closest_index < len(track_ids):
                    closest_id = track_ids[closest_index]
                    recommendations.append(closest_id)
                else:
                    app.logger.warning(f"Closest index {closest_index} out of range for track_ids. Skipping recommendation.")
            else:
                app.logger.warning(f"No tracks found in cluster {i}. Skipping recommendation.")
        user_id = self.sp.me()['id']
        playlist_description = f"SpotAI Recommendations. Based on {self.sp.playlist(selected_playlist_id)['name']}. Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        playlists = self.sp.user_playlists(user_id)['items']
        playlist_id = None
        for playlist in playlists:
            if playlist['name'] == playlist_name:
                playlist_id = playlist['id']
                self.sp.playlist_replace_items(playlist_id, [])
                self.sp.playlist_change_details(playlist_id, description=playlist_description)
                break
        if not playlist_id:
            new_playlist = self.sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)
            playlist_id = new_playlist['id']

        rec_counter = Counter()
        for x in range(num_iterations):
            random_tracks = random.sample(recommendations, min(5, len(recommendations)))
            try:
                recommended_tracks = self.sp.recommendations(seed_tracks=random_tracks, limit=100)
            except spotipy.exceptions.SpotifyException as e:
                return jsonify({'error': str(e)}), 500

            new_recommendations = [track['id'] for track in recommended_tracks['tracks'] if track['id'] not in known_tracks and track['name'] not in known_names]
            rec_counter.update(new_recommendations)

        most_common_recommendations = [track_id for track_id, count in rec_counter.most_common(100)]
        self.sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist_id, tracks=most_common_recommendations)
        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        print(f"Playlist generation took {duration:.2f} seconds")  # Print the duration
        return render_template('playlist.html', playlist_name=playlist_name, playlist_url=f"https://open.spotify.com/playlist/{playlist_id}")

spot_ai = SpotAI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri="http://localhost:5000/callback")

@app.route('/')
def index():
    if session.get('token'):
        return redirect(url_for('homepage'))
    return render_template('index.html')

@app.route('/login')
def login():
    auth_url = spot_ai.sp.auth_manager.get_authorize_url()
    print(auth_url)
    return redirect(auth_url)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token = spot_ai.sp.auth_manager.get_access_token(code)
    session['token'] = token
    user_info = spot_ai.sp.current_user()
    session['user_name'] = user_info['display_name']
    session['user_image'] = user_info['images'][0]['url'] if user_info['images'] else None
    #print("Token", token)
    return redirect(url_for('homepage'))

@app.route('/homepage')
def homepage():
    #print("session:", session)
    return render_template('homepage.html')

@app.route('/create_playlist', methods=['GET', 'POST'])
def create_playlist():
    user_id = spot_ai.sp.me()['id']
    token = spot_ai.get_token()
    if not token:
        return redirect(url_for('login'))
    if request.method == 'POST':
        playlist_name = request.form.get('playlist_name')
        return spot_ai.create_playlist(user_id, playlist_name)
    return render_template('create_playlist.html')

@app.route('/create_playlist_from_playlist', methods=['GET', 'POST'])
def create_playlist_from_playlist():
    token = spot_ai.get_token()
    if not token:
        return redirect(url_for('login'))
    if request.method == 'POST':
        selected_playlist_id = request.form.get('playlist_id')
        playlist_name = request.form.get('playlist_name')
        app.logger.info(f"Selected Playlist ID: {selected_playlist_id}")
        app.logger.info(f"Playlist Name: {playlist_name}")
        session['selected_playlist_id'] = selected_playlist_id
        session['playlist_name'] = playlist_name
        
        return redirect(url_for('playlist_creation'))
    else:
        session.pop('selected_playlist_id', None)
        session.pop('playlist_name', None)
        playlists = spot_ai.sp.current_user_playlists(limit=50)['items']
        return render_template('select_playlist.html', playlists=playlists)

@app.route('/playlist_creation')
def playlist_creation():
    user_id = spot_ai.sp.me()['id']
    selected_playlist_id = session.get('selected_playlist_id')
    playlist_name = session.get('playlist_name')

    if not selected_playlist_id or not playlist_name:
        return redirect(url_for('create_playlist_from_playlist'))
    
    response = spot_ai.create_playlist_from_playlist(user_id, selected_playlist_id, playlist_name)

    session.pop('selected_playlist_id', None)
    session.pop('playlist_name', None)

    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)
