import { useState, useEffect } from 'react';
import './SymptomChecker.css';

const SymptomChecker = () => {
  const [symptoms, setSymptoms] = useState({});
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:5000';

  useEffect(() => {
    // Fetch available symptoms from the API
    fetchSymptoms();
  }, []);

  const fetchSymptoms = async () => {
    try {
      const response = await fetch(`${API_URL}/api/symptoms`);
      const data = await response.json();
      setAvailableSymptoms(data.symptoms);
      
      // Initialize all symptoms as false
      const initialSymptoms = {};
      data.symptoms.forEach(symptom => {
        initialSymptoms[symptom] = false;
      });
      setSymptoms(initialSymptoms);
    } catch (err) {
      setError('Failed to load symptoms. Make sure Flask API is running.');
      console.error('Error fetching symptoms:', err);
    }
  };

  const handleSymptomChange = (symptom) => {
    setSymptoms(prev => ({
      ...prev,
      [symptom]: !prev[symptom]
    }));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to get prediction. Please check if the Flask API is running on port 5000.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    const resetSymptoms = {};
    availableSymptoms.forEach(symptom => {
      resetSymptoms[symptom] = false;
    });
    setSymptoms(resetSymptoms);
    setResult(null);
    setError(null);
  };

  const formatSymptomName = (symptom) => {
    return symptom.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const selectedCount = Object.values(symptoms).filter(Boolean).length;

  return (
    <div className="symptom-checker">
      <div className="container">
        <h1>Disease Prediction System</h1>
        <p className="subtitle">Select your symptoms to predict possible disease</p>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="symptoms-grid">
            {availableSymptoms.map((symptom) => (
              <label key={symptom} className={`symptom-item ${symptoms[symptom] ? 'selected' : ''}`}>
                <input
                  type="checkbox"
                  checked={symptoms[symptom] || false}
                  onChange={() => handleSymptomChange(symptom)}
                />
                <span className="symptom-label">{formatSymptomName(symptom)}</span>
              </label>
            ))}
          </div>

          <div className="actions">
            <div className="selected-count">
              Selected: {selectedCount} symptom{selectedCount !== 1 ? 's' : ''}
            </div>
            <div className="button-group">
              <button 
                type="button" 
                onClick={handleReset} 
                className="btn btn-secondary"
                disabled={loading || selectedCount === 0}
              >
                Reset
              </button>
              <button 
                type="submit" 
                className="btn btn-primary"
                disabled={loading || selectedCount === 0}
              >
                {loading ? 'Analyzing...' : 'Predict Disease'}
              </button>
            </div>
          </div>
        </form>

        {result && (
          <div className="result-card">
            <h2>Prediction Result</h2>
            <div className="result-content">
              <div className="result-item">
                <span className="result-label">Predicted Disease:</span>
                <span className="result-value disease-name">{result.disease}</span>
              </div>
              {result.confidence && (
                <div className="result-item">
                  <span className="result-label">Confidence:</span>
                  <span className="result-value">{result.confidence.toFixed(2)}%</span>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill" 
                      style={{width: `${result.confidence}%`}}
                    ></div>
                  </div>
                </div>
              )}
              <div className="result-disclaimer">
                <strong>Disclaimer:</strong> This is an AI prediction and should not replace professional medical advice. Please consult a healthcare provider for accurate diagnosis.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SymptomChecker;
