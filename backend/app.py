"""
Music Recommendation System Backend
"""
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from db import artists_data  # Import artists data from the database file

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Prepare data for content-based filtering
def prepare_data():
    # Create a DataFrame
    df = pd.DataFrame(artists_data)
    
    # Create a string representation of genres for each artist
    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))
    
    # Create TF-IDF matrix from genres
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['genres_str'])
    
    # Compute similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

# Initialize data
df, cosine_sim = prepare_data()

@app.route('/api/artists', methods=['GET'])
def get_artists():
    """Return list of all artists"""
    return jsonify(artists_data)

@app.route('/api/artist_songs/<artist_id>', methods=['GET'])
def get_artist_songs(artist_id):
    """Get songs of a selected artist"""
    artist = next((a for a in artists_data if a["id"] == artist_id), None)
    if artist:
        return jsonify({"artist": artist["name"], "songs": artist.get("songs", [])})
    else:
        return jsonify({"error": "Artist not found"}), 404

@app.route('/api/recommendations/<artist_id>', methods=['GET'])
def get_recommendations(artist_id):
    """Get artist recommendations based on similarity"""
    try:
        # Find the index of the artist in our dataframe
        artist_index = df[df['id'] == artist_id].index[0]
        
        # Get similarity scores for this artist with all others
        similarity_scores = list(enumerate(cosine_sim[artist_index]))
        
        # Sort artists based on similarity scores
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 5 similar artists (excluding itself)
        similar_artists_indices = [i[0] for i in similarity_scores[1:6]]
        
        # Create recommendations list
        recommendations = []
        for idx in similar_artists_indices:
            artist = df.iloc[idx].to_dict()
            # Remove genres_str field as it's only for internal use
            if 'genres_str' in artist:
                del artist['genres_str']
            artist['similarity'] = float(cosine_sim[artist_index][idx])
            # Make sure songs are included in recommendations
            artist['songs'] = artist.get('songs', [])
            recommendations.append(artist)
        
        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 404

# Run the application
if __name__ == '__main__':
    app.run(debug=True)