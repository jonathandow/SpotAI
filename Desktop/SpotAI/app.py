import os
from flask import Flask, render_template, redirect, request, session, jsonify
from dotenv import load_dotenv
from spotify import SpotifyClient
from ai import MusicAI
import vector_db
from flask_caching import Cache
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

spotify = SpotifyClient()
music_ai = MusicAI()

@app.route('/')
def index():
    if 'token_info' not in session:
        return render_template('index.html', authenticated=False)
    return render_template('dashboard.html', authenticated=True)

@app.route('/login')
def login():
    # Show styled login screen
    return render_template('login.html')

@app.route('/login/spotify')
def login_spotify():
    auth_url = spotify.get_auth_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if code:
        token_info = spotify.get_token(code)
        session['token_info'] = token_info
        return redirect('/dashboard')
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    if 'token_info' not in session:
        return redirect('/')
    return render_template('dashboard.html')

@app.route('/api/playlists')
def get_playlists():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    spotify.set_token(session['token_info'])
    try:
        playlists = spotify.get_user_playlists()
        return jsonify([{'id': p['id'], 'name': p['name']} for p in playlists])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlist/<playlist_id>/tracks')
def get_playlist_tracks(playlist_id):
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    spotify.set_token(session['token_info'])
    try:
        tracks = spotify.get_playlist_tracks(playlist_id)
        return jsonify([{
            'id': t['id'],
            'name': t['name'],
            'artists': [a['name'] for a in t['artists']],
            'album': t['album']['name'],
            'image': t['album']['images'][0]['url'] if t['album']['images'] else None
        } for t in tracks])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top-tracks')
def get_top_tracks():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    spotify.set_token(session['token_info'])
    try:
        tracks = spotify.get_top_tracks(limit=20)
        return jsonify([{
            'id': t['id'],
            'name': t['name'],
            'artists': [a['name'] for a in t['artists']],
            'album': t['album']['name'],
            'image': t['album']['images'][0]['url'] if t['album']['images'] else None
        } for t in tracks])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent-tracks')
def get_recent_tracks():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    spotify.set_token(session['token_info'])
    try:
        tracks = spotify.get_recent_tracks(limit=20)
        return jsonify([{
            'id': t['track']['id'],
            'name': t['track']['name'],
            'artists': [a['name'] for a in t['track']['artists']],
            'album': t['track']['album']['name'],
            'image': t['track']['album']['images'][0]['url'] if t['track']['album']['images'] else None
        } for t in tracks])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    spotify.set_token(session['token_info'])
    data = request.get_json() or {}
    seed_tracks = data.get('seed_tracks', [])
    if not seed_tracks:
        return jsonify({'error': 'No seed tracks provided', 'tracks': [], 'explanation': 'Please provide seed_tracks.'}), 400
    # Generate and upsert embeddings
    track_feats = spotify.get_track_features(seed_tracks)
    embeddings = music_ai.generate_embeddings(track_feats)
    vector_db.upsert_embeddings(seed_tracks, embeddings)
    # Query similar tracks
    centroid = np.mean(embeddings, axis=0).tolist()
    similar_ids = vector_db.query_similar(centroid)
    tracks = [spotify.get_track_info(tid) for tid in similar_ids]
    # Chat-based explanation
    summary = '\n'.join(str(f) for f in track_feats)
    explanation = music_ai.chat_recommend(summary, n=len(similar_ids))
    return jsonify({'tracks': tracks, 'explanation': explanation})

@app.route('/api/recommendations/playlist', methods=['POST'])
def get_playlist_recommendations():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    spotify.set_token(session['token_info'])
    data = request.get_json() or {}
    playlist_id = data.get('playlist_id')
    if not playlist_id:
        return jsonify({'error': 'No playlist_id provided', 'tracks': [], 'explanation': 'Please provide playlist_id.'}), 400
    # Fetch playlist tracks
    items = spotify.get_playlist_tracks(playlist_id)
    seed_ids = [t['id'] for t in items]
    if not seed_ids:
        return jsonify({'error': 'Playlist is empty', 'tracks': [], 'explanation': 'No tracks in playlist.'}), 400
    # Generate and upsert embeddings
    feats = spotify.get_track_features(seed_ids)
    embs = music_ai.generate_embeddings(feats)
    vector_db.upsert_embeddings(seed_ids, embs)
    # Query similar
    centroid = np.mean(embs, axis=0).tolist()
    sims = vector_db.query_similar(centroid)
    recs = [spotify.get_track_info(tid) for tid in sims]
    # Chat explanation
    summary = '\n'.join(str(f) for f in feats)
    explanation = music_ai.chat_recommend(summary, n=len(sims))
    return jsonify({'tracks': recs, 'explanation': explanation})

@app.route('/api/recommendations/top', methods=['POST'])
def get_top_tracks_recommendations():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    spotify.set_token(session['token_info'])
    items = spotify.get_top_tracks(limit=20)
    seed_ids = [t['id'] for t in items]
    feats = spotify.get_track_features(seed_ids)
    embs = music_ai.generate_embeddings(feats)
    vector_db.upsert_embeddings(seed_ids, embs)
    centroid = np.mean(embs, axis=0).tolist()
    sims = vector_db.query_similar(centroid)
    recs = [spotify.get_track_info(tid) for tid in sims]
    summary = '\n'.join(str(f) for f in feats)
    explanation = music_ai.chat_recommend(summary, n=len(sims))
    return jsonify({'tracks': recs, 'explanation': explanation})

@app.route('/api/recommendations/recent', methods=['POST'])
def get_recent_tracks_recommendations():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    spotify.set_token(session['token_info'])
    items = spotify.get_recent_tracks(limit=20)
    seed_ids = [t['track']['id'] for t in items]
    feats = spotify.get_track_features(seed_ids)
    embs = music_ai.generate_embeddings(feats)
    vector_db.upsert_embeddings(seed_ids, embs)
    centroid = np.mean(embs, axis=0).tolist()
    sims = vector_db.query_similar(centroid)
    recs = [spotify.get_track_info(tid) for tid in sims]
    summary = '\n'.join(str(f) for f in feats)
    explanation = music_ai.chat_recommend(summary, n=len(sims))
    return jsonify({'tracks': recs, 'explanation': explanation})

@app.route('/api/create_playlist', methods=['POST'])
def create_playlist():
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    spotify.set_token(session['token_info'])
    data = request.get_json()
    
    name = data.get('name', 'SpotAI Playlist')
    tracks = data.get('tracks', [])
    description = data.get('description', '')
    
    playlist = spotify.create_playlist(name, tracks, description)
    return jsonify(playlist)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
