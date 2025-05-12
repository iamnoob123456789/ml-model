import React, { useState } from 'react';
import './App.css';
import VideoUpload from "./components/VideoUpload";
import Results from "./components/Results";
import {useEffect} from "react";
function App() {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleResults = (data) => {
    setResults(data);
    setError(null);
  };

  const handleError = (err) => {
    setError(err.message);
    setResults(null);
  };

  return (
    <div className="App">
      <h1>Student Attentiveness Portal</h1>
      <VideoUpload onResults={handleResults} onError={handleError} />
      {error && <p className="error">Error: {error}</p>}
      {results && <Results data={results} />}
    </div>
  );
}

export default App;