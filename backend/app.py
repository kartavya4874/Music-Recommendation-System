"""
Music Recommendation System Backend
"""
from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import logging
import time
import os
from db import artists_data  # Import artists data from the database file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add configuration
app.config.update(
    DEBUG=True,
    RECOMMENDATION_CACHE_TTL=3600,  # Cache TTL in seconds (1 hour)
    MAX_RECOMMENDATIONS=10,  # Maximum number of recommendations
)

# In-memory cache for recommendations
recommendation_cache = {}
artist_cache = {}

@lru_cache(maxsize=128)
def prepare_data():
    """
    Prepare data for content-based filtering with caching
    Returns a DataFrame and cosine similarity matrix
    """
    logger.info("Preparing data for recommendations...")
    start_time = time.time()
    
    try:
        # Create a DataFrame
        df = pd.DataFrame(artists_data)
        
        # Create a string representation of genres for each artist
        df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # Create TF-IDF matrix from genres
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['genres_str'])
        
        # Compute similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        logger.info(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
        return df, cosine_sim
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        # Return empty dataframe and similarity matrix as fallback
        return pd.DataFrame(), np.array([])

# Initialize data at startup
df, cosine_sim = prepare_data()

@app.route('/api/artists', methods=['GET'])
def get_artists():
    """Return list of all artists"""
    logger.info("API call: Get all artists")
    
    try:
        # Add sorting capability
        sort_by = request.args.get('sort', default='name')
        order = request.args.get('order', default='asc')
        
        # Create a copy of the data to avoid modifying the original
        artists_copy = artists_data.copy()
        
        # Sort the data
        if sort_by in ['name', 'id']:
            reverse_order = order.lower() == 'desc'
            artists_copy.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse_order)
            
        return jsonify(artists_copy)
    except Exception as e:
        logger.error(f"Error getting artists: {str(e)}")
        return jsonify({"error": "Failed to retrieve artists"}), 500

@app.route('/api/artist_songs/<artist_id>', methods=['GET'])
def get_artist_songs(artist_id):
    """Get songs of a selected artist"""
    logger.info(f"API call: Get songs for artist ID: {artist_id}")
    
    if not artist_id or not artist_id.strip():
        return jsonify({"error": "Artist ID is required"}), 400
    
    try:
        # Check if artist exists in cache
        if artist_id in artist_cache:
            logger.info(f"Retrieved artist {artist_id} from cache")
            return jsonify(artist_cache[artist_id])
        
        # Find artist in the database
        artist = next((a for a in artists_data if a["id"] == artist_id), None)
        if not artist:
            return jsonify({"error": f"Artist with ID {artist_id} not found"}), 404
        
        # Prepare response
        response = {
            "artist": artist["name"],
            "genres": artist.get("genres", []),
            "songs": artist.get("songs", [])
        }
        
        # Store in cache
        artist_cache[artist_id] = response
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting songs for artist {artist_id}: {str(e)}")
        return jsonify({"error": f"Failed to retrieve songs for artist {artist_id}"}), 500

@app.route('/api/recommendations/<artist_id>', methods=['GET'])
def get_recommendations(artist_id):
    """Get artist recommendations based on similarity"""
    logger.info(f"API call: Get recommendations for artist ID: {artist_id}")
    
    if not artist_id or not artist_id.strip():
        return jsonify({"error": "Artist ID is required"}), 400
    
    try:
        # Check cache first for faster responses
        cache_key = f"recommendations_{artist_id}"
        if cache_key in recommendation_cache:
            cache_entry = recommendation_cache[cache_key]
            # Check if cache is still valid
            if time.time() - cache_entry["timestamp"] < app.config["RECOMMENDATION_CACHE_TTL"]:
                logger.info(f"Retrieved recommendations for artist {artist_id} from cache")
                return jsonify(cache_entry["data"])
        
        # Get number of recommendations from request or use default
        limit = request.args.get('limit', default=5, type=int)
        # Ensure limit is within bounds
        limit = min(max(1, limit), app.config["MAX_RECOMMENDATIONS"])
        
        # Find the index of the artist in our dataframe
        artist_indices = df[df['id'] == artist_id].index
        if len(artist_indices) == 0:
            return jsonify({"error": f"Artist with ID {artist_id} not found"}), 404
        
        artist_index = artist_indices[0]
        
        # Get similarity scores for this artist with all others
        similarity_scores = list(enumerate(cosine_sim[artist_index]))
        
        # Sort artists based on similarity scores
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar artists (excluding itself)
        similar_artists_indices = [i[0] for i in similarity_scores[1:limit+1]]
        
        # Create recommendations list
        recommendations = []
        for idx in similar_artists_indices:
            if idx < len(df):  # Ensure index is valid
                artist = df.iloc[idx].to_dict()
                # Remove genres_str field as it's only for internal use
                if 'genres_str' in artist:
                    del artist['genres_str']
                    
                # Add similarity score (rounded to 2 decimal places for readability)
                artist['similarity'] = round(float(cosine_sim[artist_index][idx]), 2)
                
                # Make sure songs are included in recommendations
                artist_id_to_find = artist.get('id')
                original_artist = next((a for a in artists_data if a["id"] == artist_id_to_find), None)
                if original_artist:
                    artist['songs'] = original_artist.get('songs', [])
                
                recommendations.append(artist)
        
        # Store in cache with timestamp
        recommendation_cache[cache_key] = {
            "timestamp": time.time(),
            "data": recommendations
        }
        
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting recommendations for artist {artist_id}: {str(e)}")
        return jsonify({"error": f"Failed to retrieve recommendations for artist {artist_id}"}), 500

@app.route('/api/search', methods=['GET'])
def search_artists():
    """Search for artists by name or genre"""
    logger.info("API call: Search artists")
    
    query = request.args.get('q', default='').lower()
    search_type = request.args.get('type', default='all')  # all, name, genre
    
    if not query:
        return jsonify({"error": "Search query is required"}), 400
        
    try:
        results = []
        
        for artist in artists_data:
            matched = False
            
            if search_type in ['all', 'name']:
                if query in artist['name'].lower():
                    matched = True
                    
            if not matched and search_type in ['all', 'genre']:
                for genre in artist.get('genres', []):
                    if query in genre.lower():
                        matched = True
                        break
            
            if matched:
                results.append(artist)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error searching artists: {str(e)}")
        return jsonify({"error": "Failed to search artists"}), 500

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    """API healthcheck endpoint"""
    return jsonify({
        "status": "ok",
        "version": "1.1.0",
        "timestamp": time.time()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# Function to clear caches periodically
def clear_caches():
    """Clear all caches"""
    global recommendation_cache, artist_cache
    recommendation_cache = {}
    artist_cache = {}
    prepare_data.cache_clear()
    logger.info("All caches cleared")

# Run the application
if __name__ == '__main__':
    # Initialize the application
    with app.app_context():
        # Ensure data is prepared
        df, cosine_sim = prepare_data()
        
    app.run(debug=app.config['DEBUG'], host='0.0.0.0')
