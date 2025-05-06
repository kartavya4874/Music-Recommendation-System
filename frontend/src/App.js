// src/App.js
import React, { useState, useEffect } from 'react';
import './App.css';
import ArtistSongs from './ArtistSongs';

function App() {
  const [artists, setArtists] = useState([]);
  const [selectedArtist, setSelectedArtist] = useState('');
  const [selectedArtistName, setSelectedArtistName] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [artistSongs, setArtistSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch artists when component mounts
  useEffect(() => {
    const fetchArtists = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/artists');
        if (!response.ok) {
          throw new Error('Failed to fetch artists');
        }
        const data = await response.json();
        setArtists(data);
      } catch (error) {
        setError('Failed to load artists. Please try again later.');
        console.error('Error fetching artists:', error);
      }
    };

    fetchArtists();
  }, []);

  // Filter artists based on search term
  const filteredArtists = artists.filter(artist =>
    artist.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Handle artist selection and fetch their songs
  const handleArtistClick = async (artist) => {
    setSelectedArtist(artist.id);
    setSelectedArtistName(artist.name);
    
    try {
      const response = await fetch(`http://localhost:5000/api/artist_songs/${artist.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch artist songs');
      }
      const data = await response.json();
      setArtistSongs(data.songs);
    } catch (error) {
      setError('Failed to get songs. Please try again later.');
      console.error('Error fetching songs:', error);
    }
  };

  // Handle getting recommendations
  const getRecommendations = async () => {
    if (!selectedArtist) {
      setError('Please select an artist first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:5000/api/recommendations/${selectedArtist}`);
      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }
      const data = await response.json();
      setRecommendations(data);
    } catch (error) {
      setError('Failed to get recommendations. Please try again later.');
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  // Handle clicking on a recommended artist
  const handleRecommendedArtistClick = async (artist) => {
    setSelectedArtist(artist.id);
    setSelectedArtistName(artist.name);
    setArtistSongs(artist.songs);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Raag - Bollywood & Punjabi Music Recommender</h1>
      </header>
      
      <main className="app-main">
        <section className="search-section">
          <h2>Find Artists</h2>
          
          <div className="search-container">
            <input
              type="text"
              placeholder="Search for an artist..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
          
          <div className="artists-container">
            <h3>Select an Artist:</h3>
            <div className="artists-list">
              {filteredArtists.length > 0 ? (
                filteredArtists.map(artist => (
                  <div 
                    key={artist.id} 
                    className={`artist-item ${selectedArtist === artist.id ? 'selected' : ''}`}
                    onClick={() => handleArtistClick(artist)}
                  >
                    {artist.name}
                  </div>
                ))
              ) : (
                <p>No artists found matching your search.</p>
              )}
            </div>
          </div>
          
          <button 
            className="recommendation-button"
            onClick={getRecommendations}
            disabled={!selectedArtist || loading}
          >
            {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
          </button>
          
          {error && <div className="error-message">{error}</div>}
        </section>
        
        {/* Using our new ArtistSongs component */}
        <ArtistSongs 
          artistName={selectedArtistName}
          songs={artistSongs}
        />
        
        <section className="recommendations-section">
          <h2>Recommendations</h2>
          {recommendations.length > 0 ? (
            <div className="recommendations-list">
              {recommendations.map(rec => (
                <div key={rec.id} className="recommendation-item" onClick={() => handleRecommendedArtistClick(rec)}>
                  <h3 className="recommendation-artist">{rec.name}</h3>
                  
                  {rec.genres && (
                    <p className="genres">
                      Genres: {rec.genres.join(', ')}
                    </p>
                  )}
                  
                  <p className="similarity">
                    Similarity: {(rec.similarity * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p>Select an artist and click "Get Recommendations" to see similar artists.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;