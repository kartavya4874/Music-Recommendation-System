// SongItem.js
import React, { useState } from 'react';

const SongItem = ({ title, duration, isPlaying, onClick }) => {
  const [hover, setHover] = useState(false);
  
  return (
    <div 
      className={`song-item ${isPlaying ? 'active' : ''}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={onClick}
    >
      <div className="song-title">
        {isPlaying && (
          <span style={{ color: '#1DB954', marginRight: '8px', fontWeight: 'bold' }}>
            ♪
          </span>
        )}
        {title}
      </div>
      <span className="song-duration">{duration}</span>
    </div>
  );
};

export default SongItem;