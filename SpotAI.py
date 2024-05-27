import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.cluster import KMeans

file_path = "C:/Users/dowms/SpotAI/info.txt"

with open(file_path, 'r') as f:
    CLIENT_ID = f.readline().strip()
    CLIENT_SECRET = f.readline().strip()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri="http://localhost:8888/callback",
                                               scope="user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-modify-private"))

results = sp.current_user_top_tracks(limit=50,time_range='medium_term')
recents = sp.current_user_recently_played(limit=50)

def get_all_saved_tracks(sp):
    saved = []
    results = sp.current_user_saved_tracks(limit=50)
    saved.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        saved.extend(results['items'])
    return saved

liked_songs = get_all_saved_tracks(sp)
known_tracks = {item['id'] for item in results['items']}
known_tracks.update(item['track']['id'] for item in recents['items'])
known_tracks.update(item['track']['id'] for item in liked_songs)

tracks = [{"name": track['name'], "artist": track['artists'][0]['name']} for track in results['items']]
for idx, item in enumerate(results['items']):
    print(tracks[idx]['name'] + " by " + tracks[idx]['artist'])

track_ids = [item['id'] for item in results['items']]
track_ids += (item['track']['id'] for item in recents['items'])
features = sp.audio_features(track_ids)

features_df = pd.DataFrame(features)
X = features_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

features_df['cluster'] = kmeans.labels_

print(features_df[['id', 'cluster']])

recommendations = sp.recommendations(seed_tracks=track_ids[:5], limit=100)

filtered_recommendations = [rec for rec in recommendations['tracks'] if rec['id'] not in known_tracks]
for r in filtered_recommendations[:20]:
    print(f"Recommended: {r['name']} by {r['artists'][0]['name']}")

user_id = sp.me()['id']
playlist_name = "SpotAI Recommendations"
playlist_description = "SpotAI uses a user's listening history to recommend songs they might not have heard yet."
new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)

track_uris = [song['uri'] for song in filtered_recommendations]
sp.playlist_add_items(playlist_id = new_playlist['id'], items = track_uris)

print(f"Playlist {playlist_name} has been created!")