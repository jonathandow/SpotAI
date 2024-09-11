<h1 align="center">SpotAI: Personalized Spotify Playlist Generator</h1>
<div align="center">
  
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
  ![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
  ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
  ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
  ![Bootstrap](https://img.shields.io/badge/bootstrap-%238511FA.svg?style=for-the-badge&logo=bootstrap&logoColor=white)
  ![Spotify](https://img.shields.io/badge/Spotify-1ED760?style=for-the-badge&logo=spotify&logoColor=white)
  
</div>

## About

**SpotAI** is a Flask-based web application that generates personalized Spotify playlists using machine learning algorithms. By leveraging Spotify's API and advanced clustering techniques, such as K-Means and Gaussian Mixture Models (GMM), SpotAI analyzes your listening data to create custom playlists tailored to your music preferences.

## Key Features

- **Spotify Integration**: SpotAI integrates with Spotify's API to access your top tracks, recently played songs, and saved tracks.
- **Machine Learning Models**: SpotAI uses K-Means and GMM clustering to group similar songs based on their audio features, such as danceability, energy, and tempo.
- **Playlist Generation**: Automatically creates Spotify playlists with recommended songs based on your listening habits and preferences.
- **PCA for Dimensionality Reduction**: Principal Component Analysis (PCA) is applied to optimize the clustering process by reducing the number of features.
- **Caching for Improved Performance**: SpotAI caches API responses and clustering results to enhance performance and minimize API requests.
- **Logging**: All application activity is logged to provide insights and error tracking.

## How It Works

1. **User Authentication**: Log in to your Spotify account through the application using OAuth2 authentication.
2. **Data Collection**: SpotAI retrieves your top tracks, recently played songs, and saved tracks from your Spotify library.
3. **Audio Feature Extraction**: The app extracts various audio features (e.g., danceability, energy) for the collected songs.
4. **Clustering and Recommendation**: Using K-Means and GMM clustering, SpotAI groups songs into clusters based on their features and generates song recommendations.
5. **Playlist Creation**: SpotAI creates a new Spotify playlist, populated with the top recommended tracks for you.

## Technologies Used

- **Flask**: Python web framework to serve the app.
- **Spotify API**: Used to interact with your Spotify data.
- **Spotipy**: Python client for the Spotify Web API.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: Machine learning library for clustering algorithms (K-Means, GMM) and PCA.
- **Caching**: Implemented using Flask-Caching for performance optimization.


## How to Run:

1. Clone repo
2. Add necessary keys to "info.txt"
3. run app.py with python interpreter
