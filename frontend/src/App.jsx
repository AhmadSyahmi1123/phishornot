import React, { useState } from 'react';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('https://phishornot.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url })
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Error connecting to backend.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>phishornot?</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter a URL..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Checking...' : 'Check URL'}
        </button>
      </form>

      {result && (
        <div className="result">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <p><strong>URL:</strong> {result.url}</p>
              <p><strong>Status:</strong> {result.is_phishing}</p>
              <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
