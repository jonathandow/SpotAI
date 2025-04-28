import os
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import openai
from sklearn.preprocessing import StandardScaler

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class MusicEncoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, embedding_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.encoder(x)

class MusicAI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = MusicEncoder().to(self.device)
        self.scaler = StandardScaler()
        self.text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load model weights if they exist
        if os.path.exists('model_weights/encoder.pt'):
            self.encoder.load_state_dict(torch.load('model_weights/encoder.pt'))
        self.encoder.eval()

    def _extract_features(self, track_features: List[Dict[str, Any]]) -> np.ndarray:
        """Extract relevant features from track data."""
        features = []
        for track in track_features:
            if track:
                feat = [
                    track.get('acousticness', 0),
                    track.get('danceability', 0),
                    track.get('energy', 0),
                    track.get('instrumentalness', 0),
                    track.get('valence', 0),
                    track.get('tempo', 0) / 200,  # Normalize tempo
                    track.get('loudness', 0) / -60,  # Normalize loudness
                    track.get('speechiness', 0)
                ]
                features.append(feat)
        return np.array(features)

    def generate_embeddings(self, track_features: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for tracks using OpenAI."""
        inputs = [str(f) for f in track_features]
        response = openai.Embeddings.create(
            model="text-embedding-ada-002",
            input=inputs
        )
        return [d.embedding for d in response.data]

    def get_recommendations(self, embeddings: np.ndarray, seed_tracks: List[str]) -> Dict[str, Any]:
        """Get AI-enhanced music recommendations."""
        if len(embeddings) == 0:
            return {'error': 'No valid embeddings provided'}
            
        # Calculate centroid of embeddings
        centroid = np.mean(embeddings, axis=0)
        
        # Generate text description of the musical style
        style_prompt = self._generate_style_description(centroid)
        
        # Use OpenAI to generate recommendations
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a music expert AI that provides detailed, personalized music recommendations."},
                    {"role": "user", "content": f"Based on this musical style: {style_prompt}, suggest specific musical characteristics (danceability, energy, valence, etc.) that would make good recommendations. Focus on the audio features, not specific artists or songs."}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            # Extract feature targets from the AI response
            ai_suggestions = response.choices[0].message.content
            feature_targets = self._parse_ai_suggestions(ai_suggestions)
            
            return {
                'feature_targets': feature_targets,
                'explanation': ai_suggestions
            }
            
        except Exception as e:
            print(f"Error getting AI recommendations: {str(e)}")
            # Fallback to basic feature matching
            return {
                'feature_targets': [{
                    'danceability': float(np.random.normal(0.5, 0.1)),
                    'energy': float(np.random.normal(0.5, 0.1)),
                    'valence': float(np.random.normal(0.5, 0.1)),
                    'acousticness': float(np.random.normal(0.5, 0.1)),
                    'instrumentalness': float(np.random.normal(0.5, 0.1))
                }],
                'explanation': "Based on your music taste, looking for similar energetic and rhythmic patterns."
            }

    def _generate_style_description(self, embedding: np.ndarray) -> str:
        """Generate a text description of the musical style from embeddings."""
        # Map embedding dimensions to musical characteristics
        energy = (embedding[0] + 1) / 2  # Convert from [-1,1] to [0,1]
        complexity = (embedding[1] + 1) / 2
        mood = (embedding[2] + 1) / 2
        
        style = []
        if energy > 0.7:
            style.append("high-energy")
        elif energy < 0.3:
            style.append("calm and relaxed")
            
        if complexity > 0.7:
            style.append("complex and intricate")
        elif complexity < 0.3:
            style.append("straightforward and accessible")
            
        if mood > 0.7:
            style.append("upbeat and positive")
        elif mood < 0.3:
            style.append("melancholic and introspective")
            
        return f"A {', '.join(style)} musical style"

    def _parse_ai_suggestions(self, ai_response: str) -> List[Dict[str, float]]:
        """Parse AI response into feature targets."""
        # Default features if parsing fails
        default_features = {
            'danceability': 0.5,
            'energy': 0.5,
            'valence': 0.5,
            'acousticness': 0.5,
            'instrumentalness': 0.5
        }
        
        try:
            # Extract numerical values and keywords from the AI response
            features = default_features.copy()
            
            if "high energy" in ai_response.lower():
                features['energy'] = 0.8
            elif "low energy" in ai_response.lower():
                features['energy'] = 0.2
                
            if "danceable" in ai_response.lower():
                features['danceability'] = 0.8
            elif "less danceable" in ai_response.lower():
                features['danceability'] = 0.3
                
            if "happy" in ai_response.lower() or "positive" in ai_response.lower():
                features['valence'] = 0.8
            elif "sad" in ai_response.lower() or "melancholic" in ai_response.lower():
                features['valence'] = 0.2
                
            if "acoustic" in ai_response.lower():
                features['acousticness'] = 0.8
            elif "electronic" in ai_response.lower():
                features['acousticness'] = 0.2
                
            return [features]
            
        except Exception as e:
            print(f"Error parsing AI suggestions: {str(e)}")
            return [default_features]

    def chat_recommend(self, seed_summary: str, n: int = 10) -> str:
        """Use OpenAI chat model to recommend songs and explanations."""
        prompt = (
            f"I have these seed tracks with their features:\n{seed_summary}\n\n"
            f"Recommend {n} more songs from Spotify’s catalog, and explain in one sentence per track why."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300
        )
        return resp.choices[0].message.content
