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
        # Store the refreshed token back to session when needed
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
    """Get AI recommendations based on chosen playlists."""
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    spotify.set_token(session['token_info'])
    # Update session if token was refreshed
    try:
        if hasattr(spotify, 'refreshed_token') and spotify.refreshed_token:
            session['token_info'] = spotify.refreshed_token
            spotify.refreshed_token = None
    except Exception:
        pass
    
    data = request.get_json()
    playlist_id = data.get('playlist_id', '')
    
    try:
        # Get tracks from the playlist
        items = spotify.get_playlist_tracks(playlist_id)
        if not items:
            return jsonify({'error': 'No tracks found in playlist'}), 404
        
        # Extract track IDs
        track_ids = [item['id'] for item in items if item.get('id')]
        seed_ids = track_ids[:10]  # Use first 10 tracks as seeds
        
        # TEMPORARY FALLBACK: Return tracks from the same playlist
        print("Using fallback: Returning tracks from the same playlist")
        
        # Just use the tracks from the playlist that weren't used as seeds
        recommended_tracks = [
            {
                'id': item['id'],
                'name': item['name'],
                'artists': [a['name'] for a in item['artists']],
                'album': item['album']['name'],
                'image': item['album']['images'][0]['url'] if item['album']['images'] else None,
                'uri': item['uri']
            }
            for item in items[10:30] if item.get('id') not in seed_ids
        ]
        
        if not recommended_tracks:
            # If we don't have enough tracks, just return all tracks
            recommended_tracks = [
                {
                    'id': item['id'],
                    'name': item['name'],
                    'artists': [a['name'] for a in item['artists']],
                    'album': item['album']['name'],
                    'image': item['album']['images'][0]['url'] if item['album']['images'] else None,
                    'uri': item['uri']
                }
                for item in items[:20]
            ]
            
        return jsonify({
            'tracks': recommended_tracks,
            'message': 'More tracks from your playlist'
        })
        
        try:
            # Get audio features for the tracks
            feats = spotify.get_track_features(track_ids)
            if not feats:
                return jsonify({'error': 'Could not fetch audio features'}), 500
            
            # Generate embeddings
            embs = music_ai.generate_embeddings(feats)
            if not embs:
                return jsonify({'error': 'Failed to generate embeddings'}), 500
            
            try:
                # Save embeddings to vector DB
                vector_db.upsert_embeddings(seed_ids, embs)
            except Exception as e:
                print(f"Error upserting to vector DB: {e}")
                # Continue even if vector DB fails
            
            # Get AI recommendations
            rec_data = music_ai.get_recommendations(embs, seed_ids)
            
            # Check for errors in recommendation generation
            if 'error' in rec_data:
                return jsonify({'error': rec_data['error']}), 500
            
            # Find similar tracks
            centroid = rec_data.get('centroid')
            
            try:
                sims = vector_db.query_similar(centroid)
            except Exception as e:
                print(f"Error querying vector DB: {e}")
                sims = []
            
            # Get track details for recommendations
            recommended_tracks = []
            if sims:
                recommended_tracks = spotify.get_tracks(sims)
            
            # If vector DB recommendations failed, use Spotify recommendations as fallback
            if not recommended_tracks:
                print("Falling back to Spotify recommendations")
                recommended_tracks = spotify.get_recommendations(seed_ids[:5], limit=20)
            
            return jsonify({
                'tracks': recommended_tracks,
                'message': 'AI-powered recommendations based on your playlist'
            })
            
        except Exception as e:
            print(f"Error in recommendation pipeline: {e}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error fetching playlist: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/top', methods=['POST'])
def get_top_tracks_recommendations():
    """Get AI recommendations based on user's top tracks."""
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    spotify.set_token(session['token_info'])
    
    try:
        # Get user's top tracks
        items = spotify.get_top_tracks(limit=50)  # Get more tracks to use some as "recommendations"
        if not items:
            return jsonify({'error': 'No top tracks found'}), 404
            
        # Use first 10 as seeds, the rest as "recommendations"
        seed_ids = [item['id'] for item in items[:10] if item.get('id')]
        
        # TEMPORARY FALLBACK: Use remaining top tracks as recommendations
        print("Using fallback: Using lower-ranked top tracks as recommendations")
        
        # Format the remaining tracks as recommendations
        recommended_tracks = [
            {
                'id': item['id'],
                'name': item['name'],
                'artists': [a['name'] for a in item['artists']],
                'album': item['album']['name'],
                'image': item['album']['images'][0]['url'] if item['album']['images'] else None,
                'uri': item['uri']
            }
            for item in items[10:30] if item.get('id') not in seed_ids
        ]
        
        return jsonify({
            'tracks': recommended_tracks,
            'message': 'More tracks you might enjoy based on your listening history'
        })
        
        try:
            # Get audio features for the tracks
            feats = spotify.get_track_features(track_ids)
            if not feats:
                return jsonify({'error': 'Could not fetch audio features'}), 500
                
            # Generate embeddings
            embs = music_ai.generate_embeddings(feats)
            if not embs:
                return jsonify({'error': 'Failed to generate embeddings'}), 500
                
            try:
                # Save embeddings to vector DB
                vector_db.upsert_embeddings(seed_ids, embs)
            except Exception as e:
                print(f"Error upserting to vector DB: {e}")
                # Continue even if vector DB fails
                
            # Get AI recommendations
            rec_data = music_ai.get_recommendations(embs, seed_ids)
            
            # Check for errors in recommendation generation
            if 'error' in rec_data:
                return jsonify({'error': rec_data['error']}), 500
                
            # Find similar tracks
            centroid = rec_data.get('centroid')
            
            try:
                sims = vector_db.query_similar(centroid)
            except Exception as e:
                print(f"Error querying vector DB: {e}")
                sims = []
                
            # Get track details for recommendations
            recommended_tracks = []
            if sims:
                recommended_tracks = spotify.get_tracks(sims)
                
            # If vector DB recommendations failed, use Spotify recommendations as fallback
            if not recommended_tracks:
                print("Falling back to Spotify recommendations")
                recommended_tracks = spotify.get_recommendations(seed_ids[:5], limit=20)
                
            return jsonify({
                'tracks': recommended_tracks,
                'message': 'AI-powered recommendations based on your top tracks'
            })
            
        except Exception as e:
            print(f"Error in recommendation pipeline: {e}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error fetching top tracks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/recent', methods=['POST'])
def get_recent_tracks_recommendations():
    """Get AI recommendations based on user's recently played tracks."""
    if 'token_info' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    spotify.set_token(session['token_info'])
    
    try:
        # Get user's recently played tracks
        items = spotify.get_recent_tracks(limit=50)  # Get more to use as recommendations
        if not items:
            return jsonify({'error': 'No recent tracks found'}), 404
            
        # Use first items as seeds
        recent_tracks = [item['track'] for item in items if item.get('track')]
        first_10_ids = [track['id'] for track in recent_tracks[:10] if track.get('id')]
        
        # TEMPORARY FALLBACK: Use remaining recent tracks as recommendations
        print("Using fallback: Using older recent tracks as recommendations")
        
        # Format the remaining tracks as recommendations
        remaining_tracks = recent_tracks[10:30]
        recommended_tracks = [
            {
                'id': item['id'],
                'name': item['name'],
                'artists': [a['name'] for a in item['artists']],
                'album': item['album']['name'],
                'image': item['album']['images'][0]['url'] if item['album']['images'] else None,
                'uri': item['uri']
            }
            for item in remaining_tracks if item.get('id') not in first_10_ids
        ]
        
        return jsonify({
            'tracks': recommended_tracks,
            'message': 'More tracks from your recent listening history'
        })
        
        try:
            # Get audio features for the tracks
            feats = spotify.get_track_features(track_ids)
            if not feats:
                return jsonify({'error': 'Could not fetch audio features'}), 500
                
            # Generate embeddings
            embs = music_ai.generate_embeddings(feats)
            if not embs:
                return jsonify({'error': 'Failed to generate embeddings'}), 500
                
            try:
                # Save embeddings to vector DB
                vector_db.upsert_embeddings(seed_ids, embs)
            except Exception as e:
                print(f"Error upserting to vector DB: {e}")
                # Continue even if vector DB fails
                
            # Get AI recommendations
            rec_data = music_ai.get_recommendations(embs, seed_ids)
            
            # Check for errors in recommendation generation
            if 'error' in rec_data:
                return jsonify({'error': rec_data['error']}), 500
                
            # Find similar tracks
            centroid = rec_data.get('centroid')
            
            try:
                sims = vector_db.query_similar(centroid)
            except Exception as e:
                print(f"Error querying vector DB: {e}")
                sims = []
                
            # Get track details for recommendations
            recommended_tracks = []
            if sims:
                recommended_tracks = spotify.get_tracks(sims)
                
            # If vector DB recommendations failed, use Spotify recommendations as fallback
            if not recommended_tracks:
                print("Falling back to Spotify recommendations")
                recommended_tracks = spotify.get_recommendations(seed_ids[:5], limit=20)
                
            return jsonify({
                'tracks': recommended_tracks,
                'message': 'AI-powered recommendations based on your recent listening history'
            })
            
        except Exception as e:
            print(f"Error in recommendation pipeline: {e}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error fetching recent tracks: {e}")
        return jsonify({'error': str(e)}), 500

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
