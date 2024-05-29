import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.cluster import KMeans
from flask import Flask, redirect, request, session, url_for, render_template, jsonify
from tenacity import retry, wait_exponential, stop_after_attempt
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_COOKIE_NAME'] = 'SpotAI'

file_path = "C:/Users/dowms/SpotAI/info.txt"

with open(file_path, 'r') as f:
    CLIENT_ID = f.readline().strip()
    CLIENT_SECRET = f.readline().strip()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri="http://localhost:5000/callback",
                                               scope="user-library-read user-read-recently-played user-top-read playlist-modify-public"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    auth_url = sp.auth_manager.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token = sp.auth_manager.get_access_token(code)
    session['token'] = token
    return redirect(url_for('create_playlist'))

def get_token():
    token = session.get('token', None)
    if not token:
        return None
    if sp.auth_manager.is_token_expired(token):
        token = sp.auth_manager.refresh_token(token['refresh_token'])
    return token

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def get_audio_features_for_tracks(sp, track_ids):
    return sp.audio_features(track_ids)

def get_all_top_tracks(sp):
    top = []
    results = sp.current_user_top_tracks(limit=50, time_range='medium_term')
    top.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        top.extend(results['items'])
    return top

def get_recent_tracks(sp):
    recents = []
    results = sp.current_user_recently_played(limit=50)
    recents.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        recents.extend(results['items'])
    return recents

def get_all_saved_tracks(sp):
    saved = []
    results = sp.current_user_saved_tracks(limit=50)
    saved.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        saved.extend(results['items'])
    return saved

@app.route('/create_playlist', methods=['GET', 'POST'])
def create_playlist():
    token = get_token()
    if not token:
        return redirect(url_for('login'))
    if request.method == 'POST':
        playlist_name = request.form.get('playlist_name')
        num_clusters = 100
        num_iterations = 10

        top_artists = sp.current_user_top_artists(limit=5, time_range='medium_term')['items']
        artist_ids = [a['id'] for a in top_artists]

        liked_songs = get_all_saved_tracks(sp)
        recent_songs = get_recent_tracks(sp)
        top_songs = get_all_top_tracks(sp)

        known_tracks = {item['track']['id'] for item in liked_songs}
        known_tracks.update({item['track']['id'] for item in recent_songs})
        known_tracks.update({item['id'] for item in top_songs})

        track_ids = [track['id'] for track in top_songs]
        
        try:
            features = get_audio_features_for_tracks(sp, track_ids[:100])
        except spotipy.exceptions.SpotifyException as e:
            return jsonify({'error': str(e)}), 500

        if not features:
            return jsonify({'error': "No audio features retrieved for the tracks"}), 500

        features_df = pd.DataFrame(features)

        X = features_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        features_df['cluster'] = kmeans.labels_

        centroids = kmeans.cluster_centers_

        recommendations = []
        for i in range(num_clusters):
            centroid = centroids[i]
            cluster_tracks = features_df[features_df['cluster'] == i]
            closest_index = ((cluster_tracks[X.columns] - centroid) ** 2).sum(axis=1).idxmin()
            closest_id = top_songs[closest_index]['id']
            recommendations.append(closest_id)

        user_id = sp.me()['id']
        playlist_description = f"SpotAI Recommendations. Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        playlists = sp.user_playlists(user_id)['items']
        playlist_id = None
        for playlist in playlists:
            if playlist['name'] == playlist_name:
                playlist_id = playlist['id']
                sp.playlist_replace_items(playlist_id, [])
                sp.playlist_change_details(playlist_id, description=playlist_description)
                break
        if not playlist_id:
            new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)
            playlist_id = new_playlist['id']
        
        try:
            recommended_tracks = sp.recommendations(seed_tracks=recommendations[:5], limit=100)
        except spotipy.exceptions.SpotifyException as e:
            return jsonify({'error': str(e)}), 500
        
        new_recommendations = [track['id'] for track in recommended_tracks['tracks'] if track['id'] not in known_tracks]

        sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist_id, tracks=new_recommendations)

        return render_template('playlist.html', playlist_name=playlist_name, playlist_url=f"https://open.spotify.com/playlist/{playlist_id}")
    
    return render_template('create_playlist.html')


if __name__ == '__main__':
    app.run(debug=True)
