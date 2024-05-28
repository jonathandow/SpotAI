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
    CLIENT_ID = f.readline()
    CLIENT_SECRET = f.readline()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="1af6f7d6c2314be082621ca4785cda66",
                                               client_secret="5be950690dc34db0b9941109b6ae6191",
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

@app.route('/create_playlist', methods=['GET', 'POST'])
def create_playlist():
    token = get_token()
    if not token:
        return redirect(url_for('login'))
    if request.method == 'POST':
        playlist_name = request.form.get('playlist_name')
        genre = request.form.get('genre')
        num_clusters = int(request.form.get('num_clusters', '5'))

        top_artists = sp.current_user_top_artists(limit=5, time_range='medium_term')['items']
        artist_ids = [a['id'] for a in top_artists]
        artist_genres = [g for a in top_artists for g in a['genres']]

        recommendations = []
        rec_tracks = {item['id'] for item in sp.current_user_top_tracks(limit=50, time_range='medium_term')['items']}
        def get_all_saved_tracks(sp):
            saved = []
            results = sp.current_user_saved_tracks(limit=50)
            saved.extend(results['items'])
            while results['next']:
                results = sp.next(results)
                saved.extend(results['items'])
            return saved
        liked_songs = get_all_saved_tracks(sp)
        rec_tracks.update({item['track']['id'] for item in liked_songs})
        try:
            recommended_tracks = sp.recommendations(seed_tracks=list(rec_tracks)[:1], seed_artists = artist_ids[:2], seed_genres = artist_genres[:2], limit=100)
        except spotipy.exceptions.SpotifyException as e:
            return jsonify({'error': str(e)}), 500

        track_ids = [track['id'] for track in recommended_tracks['tracks']]
        count = 0
        for track in recommended_tracks['tracks']:
            count += 1
            print(track)
        print(count)

        try:
            features = get_audio_features_for_tracks(sp, track_ids)
        except spotipy.exceptions.SpotifyException as e:
            return jsonify({'error': str(e)}), 500

        features_df = pd.DataFrame(features)
        X = features_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        features_df['cluster'] = kmeans.labels_

        centroids = kmeans.cluster_centers_

        for i in range(num_clusters):
            centroid = centroids[i]
            cluster_tracks = features_df[features_df['cluster'] == i]
            closest_index = ((cluster_tracks[X.columns] - centroid) ** 2).sum(axis=1).idxmin()
            closest_id = recommended_tracks['tracks'][closest_index]['id']
            recommendations.append(closest_id)

        user_id = sp.me()['id']
        playlist_description = f"SpotAI recommends {playlist_name} based on your preference for {genre}. Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        playlists = sp.user_playlists(user_id)['items']
        playlist_id = None
        for playlist in playlists:
            if playlist['name'] == playlist_name:
                playlist_id = playlist['id']
                break
        if not playlist_id:
            new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)
            playlist_id = new_playlist['id']
        c = 0
        for r in recommendations:
            c += 1
        print("Recs: \n")
        print(c)
        uris = [song['uri'] for song in recommended_tracks['tracks']]
        sp.playlist_add_items(playlist_id = playlist_id, items = uris)

        return render_template('playlist.html', playlist_name=playlist_name, playlist_url=f"https://open.spotify.com/playlist/{playlist_id}")
    
    return render_template('create_playlist.html')

if __name__ == '__main__':
    app.run(debug=True)