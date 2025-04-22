import os
from flask import Flask, render_template, redirect, request, session, jsonify
from dotenv import load_dotenv
from spotify import SpotifyClient
from ai import MusicAI
from flask_caching import Cache

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

# Original recommendations endpoint for manually selected tracks
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        if 'token_info' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        spotify.set_token(session['token_info'])
        data = request.get_json()
        seed_tracks = data.get('seed_tracks', [])
        
        if not seed_tracks:
            return jsonify({
                'error': 'No seed tracks provided',
                'tracks': [],
                'explanation': 'Please select at least one track to get recommendations.'
            }), 400
        
        try:
            # Get track features and generate embeddings
            track_features = spotify.get_track_features(seed_tracks)
            
            # If we couldn't get features, use a simpler approach
            if not track_features:
                # Fallback to Spotify's recommendation engine directly
                tracks = spotify.sp.recommendations(seed_tracks=seed_tracks[:5], limit=20)['tracks']
                explanation = "Based on your selected tracks, we've found some songs you might enjoy."
                return jsonify({
                    'tracks': tracks,
                    'explanation': explanation
                })
            
            # Continue with AI-enhanced recommendations if we have features
            embeddings = music_ai.generate_embeddings(track_features)
            recommendations = music_ai.get_recommendations(embeddings, seed_tracks)
            
            # Get Spotify tracks based on AI recommendations
            all_tracks = spotify.get_tracks_by_features(recommendations)
            
            # Filter out recently played tracks
            filtered_tracks = []
            for track in all_tracks:
                if track and 'id' in track and not spotify.is_track_recently_played(track['id']):
                    filtered_tracks.append(track)
                if len(filtered_tracks) >= 20:  # Limit to 20 tracks
                    break
            
            # If we have too few tracks after filtering, get more recommendations
            attempts = 0
            while len(filtered_tracks) < 10 and attempts < 3:
                more_tracks = spotify.get_tracks_by_features(recommendations, limit=10)
                for track in more_tracks:
                    if track and 'id' in track and not spotify.is_track_recently_played(track['id']):
                        if track not in filtered_tracks:  # Avoid duplicates
                            filtered_tracks.append(track)
                    if len(filtered_tracks) >= 20:
                        break
                attempts += 1
                
            # If we still don't have enough tracks, just use what we have
            if not filtered_tracks:
                # Last resort: just get recommendations without filtering
                filtered_tracks = spotify.sp.recommendations(seed_tracks=seed_tracks[:5], limit=20)['tracks']
    
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        # Get the explanation from recommendations or use a default one
        explanation = recommendations.get('explanation', "Based on your selected tracks, we've found some songs you might enjoy.")
        
        return jsonify({
            'tracks': filtered_tracks,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New endpoint for playlist-based recommendations
@app.route('/api/recommendations/playlist', methods=['POST'])
def get_playlist_recommendations():
    try:
        if 'token_info' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        spotify.set_token(session['token_info'])
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        
        if not playlist_id:
            return jsonify({'error': 'No playlist ID provided'}), 400
        
        # Get all tracks from the playlist
        playlist_tracks = spotify.get_playlist_tracks(playlist_id)
        if not playlist_tracks:
            return jsonify({'error': 'No tracks found in playlist'}), 404
        
        # Extract track IDs
        track_ids = [track['id'] for track in playlist_tracks]
        
        # Use up to 5 random tracks as seeds
        import random
        seed_tracks = random.sample(track_ids, min(5, len(track_ids)))
        # Validate seed tracks
        valid_seeds = spotify.filter_valid_track_ids(seed_tracks)
        if not valid_seeds:
            return jsonify({'error': 'No valid tracks found in playlist for recommendations.'}), 400
        try:
            # Get track features and generate embeddings
            track_features = spotify.get_track_features(valid_seeds)
            # If we couldn't get features, use a simpler approach
            if not track_features:
                tracks = spotify.sp.recommendations(seed_tracks=valid_seeds[:5], limit=20)['tracks']
                explanation = f"Based on tracks from your playlist, we've found some songs you might enjoy."
                return jsonify({
                    'tracks': tracks,
                    'explanation': explanation
                })
            # Continue with AI-enhanced recommendations
            embeddings = music_ai.generate_embeddings(track_features)
            # Pass valid seeds to get_tracks_by_features
            recommendations = music_ai.get_recommendations(embeddings, valid_seeds)
            for rec in recommendations:
                rec['seed_tracks'] = valid_seeds[:5]
            all_tracks = spotify.get_tracks_by_features(recommendations)
            # Filter out tracks that are already in the playlist and recently played tracks
            filtered_tracks = []
            for track in all_tracks:
                if track and 'id' in track and track['id'] not in track_ids and not spotify.is_track_recently_played(track['id']):
                    filtered_tracks.append(track)
                if len(filtered_tracks) >= 20:
                    break
            # If we have too few tracks, get more recommendations
            if len(filtered_tracks) < 10:
                more_tracks = spotify.sp.recommendations(seed_tracks=valid_seeds[:5], limit=20)['tracks']
                for track in more_tracks:
                    if track['id'] not in track_ids and not spotify.is_track_recently_played(track['id']):
                        if not any(t['id'] == track['id'] for t in filtered_tracks):
                            filtered_tracks.append(track)
                    if len(filtered_tracks) >= 20:
                        break
        except Exception as e:
            print(f"Error in recommendation generation: {e}")
            # Fallback to simple recommendations
            filtered_tracks = spotify.sp.recommendations(seed_tracks=valid_seeds[:5], limit=20)['tracks']
        # Get playlist name for better explanation
        playlist_name = spotify.sp.playlist(playlist_id)['name']
        explanation = f"Based on your playlist '{playlist_name}', we've found some new tracks you might enjoy."
        return jsonify({
            'tracks': filtered_tracks,
            'explanation': explanation
        })
    except Exception as e:
        print(f"Error in playlist recommendations: {e}")
        return jsonify({'error': str(e)}), 500

# New endpoint for top tracks recommendations
@app.route('/api/recommendations/top', methods=['POST'])
def get_top_tracks_recommendations():
    try:
        if 'token_info' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        spotify.set_token(session['token_info'])
        
        # Get user's top tracks
        top_tracks = spotify.get_top_tracks(limit=20)
        if not top_tracks:
            return jsonify({'error': 'No top tracks found'}), 404
        
        # Extract track IDs
        track_ids = [track['id'] for track in top_tracks]
        
        # Use up to 5 tracks as seeds
        seed_tracks = track_ids[:5]
        
        try:
            # Get track features and generate embeddings
            track_features = spotify.get_track_features(seed_tracks)
            
            # If we couldn't get features, use a simpler approach
            if not track_features:
                # Fallback to Spotify's recommendation engine directly
                tracks = spotify.sp.recommendations(seed_tracks=seed_tracks, limit=20)['tracks']
                explanation = "Based on your top tracks, we've found some songs you might enjoy."
                return jsonify({
                    'tracks': tracks,
                    'explanation': explanation
                })
            
            # Continue with AI-enhanced recommendations
            embeddings = music_ai.generate_embeddings(track_features)
            recommendations = music_ai.get_recommendations(embeddings, seed_tracks)
            
            # Get Spotify tracks based on AI recommendations
            all_tracks = spotify.get_tracks_by_features(recommendations)
            
            # Filter out tracks that are already in top tracks and recently played tracks
            filtered_tracks = []
            for track in all_tracks:
                if track and 'id' in track and track['id'] not in track_ids and not spotify.is_track_recently_played(track['id']):
                    filtered_tracks.append(track)
                if len(filtered_tracks) >= 20:  # Limit to 20 tracks
                    break
        
        except Exception as e:
            print(f"Error in top tracks recommendation generation: {e}")
            # Fallback to simple recommendations
            filtered_tracks = spotify.sp.recommendations(seed_tracks=seed_tracks, limit=20)['tracks']
        
        explanation = "Based on your most played tracks, we've found some new music you might enjoy."
        
        return jsonify({
            'tracks': filtered_tracks,
            'explanation': explanation
        })
    except Exception as e:
        print(f"Error in top tracks recommendations: {e}")
        return jsonify({'error': str(e)}), 500

# New endpoint for recent tracks recommendations
@app.route('/api/recommendations/recent', methods=['POST'])
def get_recent_tracks_recommendations():
    try:
        if 'token_info' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        spotify.set_token(session['token_info'])
        
        # Get user's recently played tracks
        recent_tracks = spotify.get_recent_tracks(limit=50)
        if not recent_tracks:
            return jsonify({'error': 'No recent tracks found'}), 404
        
        # Extract track IDs
        track_ids = [item['track']['id'] for item in recent_tracks]
        
        # Use up to 5 tracks as seeds
        import random
        seed_tracks = random.sample(track_ids, min(5, len(track_ids)))
        
        try:
            # Get track features and generate embeddings
            track_features = spotify.get_track_features(seed_tracks)
            
            # If we couldn't get features, use a simpler approach
            if not track_features:
                # Fallback to Spotify's recommendation engine directly
                tracks = spotify.sp.recommendations(seed_tracks=seed_tracks, limit=20)['tracks']
                explanation = "Based on your recent listening history, we've found some songs you might enjoy."
                return jsonify({
                    'tracks': tracks,
                    'explanation': explanation
                })
            
            # Continue with AI-enhanced recommendations
            embeddings = music_ai.generate_embeddings(track_features)
            recommendations = music_ai.get_recommendations(embeddings, seed_tracks)
            
            # Get Spotify tracks based on AI recommendations
            all_tracks = spotify.get_tracks_by_features(recommendations)
            
            # Filter out tracks that are already in recent tracks
            filtered_tracks = []
            for track in all_tracks:
                if track and 'id' in track and track['id'] not in track_ids:
                    filtered_tracks.append(track)
                if len(filtered_tracks) >= 20:  # Limit to 20 tracks
                    break
        
        except Exception as e:
            print(f"Error in recent tracks recommendation generation: {e}")
            # Fallback to simple recommendations
            filtered_tracks = spotify.sp.recommendations(seed_tracks=seed_tracks, limit=20)['tracks']
        
        explanation = "Based on your recent listening history, we've curated some fresh tracks you might enjoy."
        
        return jsonify({
            'tracks': filtered_tracks,
            'explanation': explanation
        })
    except Exception as e:
        print(f"Error in recent tracks recommendations: {e}")
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
