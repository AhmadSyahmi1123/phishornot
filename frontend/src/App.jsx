import React, { useState } from 'react';
import './App.css';

function Modal({ isOpen, onClose, children }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal">
        <button className="close-button" onClick={onClose}>×</button>
        {children}
      </div>
    </div>
  );
}

function App() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setShowModal(true);

    try {
      const response = await fetch('https://phishornot.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Error connecting to backend.' });
    } finally {
      setLoading(false);
    }
  };

  const closeModal = () => {
    setShowModal(false);
    setResult(null);
    setUrl('');
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
          Check URL
        </button>
      </form>

      <Modal isOpen={showModal} onClose={closeModal}>
        {loading ? (
          <div className="spinner"></div>
        ) : result ? (
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
        ) : null}
      </Modal>
    </div>
  );
}

export default App;
