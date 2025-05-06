import React, { useState } from 'react';
import SongItem from './SongItem';

const ArtistSongs = ({ artistName, songs }) => {
  const [currentPlayingSong, setCurrentPlayingSong] = useState(null);

  const handleSongClick = (songId) => {
    setCurrentPlayingSong(songId === currentPlayingSong ? null : songId);
    // Here you could add audio playing functionality in the future
  };

  return (
    <section className="artist-songs-section">
      <h2>{artistName ? `${artistName}'s Songs` : 'Select an Artist to See Songs'}</h2>
      {songs && songs.length > 0 ? (
        <div className="songs-list">
          {songs.map(song => (
            <SongItem
              key={song.id}
              title={song.title}
              duration={song.duration}
              isPlaying={currentPlayingSong === song.id}
              onClick={() => handleSongClick(song.id)}
            />
          ))}
        </div>
      ) : (
        <p>{artistName ? 'No songs found for this artist.' : 'Click on an artist to see their songs.'}</p>
      )}
    </section>
  );
};

export default ArtistSongs;