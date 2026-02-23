import { useState } from 'react';
import { compareResponses } from '../services/api';

function CompareResponses() {
    const [prompt, setPrompt] = useState('');
    const [responses, setResponses] = useState(['', '']);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const addResponse = () => {
        if (responses.length < 10) {
            setResponses([...responses, '']);
        }
    };

    const removeResponse = (index) => {
        if (responses.length > 2) {
            setResponses(responses.filter((_, i) => i !== index));
        }
    };

    const updateResponse = (index, value) => {
        const newResponses = [...responses];
        newResponses[index] = value;
        setResponses(newResponses);
    };

    const handleCompare = async () => {
        const validResponses = responses.filter(r => r.trim());

        if (!prompt.trim() || validResponses.length < 2) {
            setError('Please enter a prompt and at least 2 responses');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const data = await compareResponses(prompt, validResponses);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to compare responses');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="compare-responses">
            <div className="card">
                <div className="card-header">
                    <h2>🔄 Compare Responses</h2>
                    <p>Compare multiple AI responses to the same prompt</p>
                </div>

                <div className="form-group">
                    <label htmlFor="compare-prompt">
                        Prompt
                        <span className="char-count">{prompt.length} characters</span>
                    </label>
                    <textarea
                        id="compare-prompt"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Enter your prompt here..."
                        rows={3}
                        disabled={loading}
                    />
                </div>

                <div className="responses-section">
                    <h3>Responses to Compare</h3>
                    {responses.map((response, index) => (
                        <div key={index} className="response-item">
                            <div className="response-header">
                                <span className="response-label">Response {index + 1}</span>
                                {responses.length > 2 && (
                                    <button
                                        className="btn-remove"
                                        onClick={() => removeResponse(index)}
                                        disabled={loading}
                                    >
                                        ✕
                                    </button>
                                )}
                            </div>
                            <textarea
                                value={response}
                                onChange={(e) => updateResponse(index, e.target.value)}
                                placeholder={`Enter response ${index + 1}...`}
                                rows={3}
                                disabled={loading}
                            />
                        </div>
                    ))}
                </div>

                <div className="button-group">
                    <button
                        className="btn btn-secondary"
                        onClick={addResponse}
                        disabled={loading || responses.length >= 10}
                    >
                        + Add Response
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleCompare}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <span className="spinner"></span>
                                Comparing...
                            </>
                        ) : (
                            `Compare ${responses.filter(r => r.trim()).length} Responses`
                        )}
                    </button>
                </div>

                {error && (
                    <div className="error-message">
                        <span className="error-icon">⚠️</span>
                        {error}
                    </div>
                )}
            </div>

            {result && (
                <div className="card">
                    <div className="card-header">
                        <h2>🏆 Comparison Results</h2>
                        <p>Ranked by score (highest to lowest)</p>
                    </div>

                    <div className="comparison-grid">
                        {result.ranked_results?.map((item, index) => (
                            <div
                                key={index}
                                className={`comparison-card ${index === 0 ? 'best' : ''}`}
                            >
                                <div className="comparison-rank">
                                    {index === 0 && <span className="crown">👑</span>}
                                    <span className="rank-number">#{index + 1}</span>
                                </div>
                                <div className="comparison-score">
                                    <div className="score-large">{item.score.toFixed(2)}</div>
                                    <div className="score-label">/5.0</div>
                                </div>
                                <div className="comparison-details">
                                    <div className="detail-row">
                                        <span className="label">Quality:</span>
                                        <span className={`quality-badge ${item.quality.toLowerCase()}`}>
                                            {item.quality}
                                        </span>
                                    </div>
                                    <div className="detail-row">
                                        <span className="label">Confidence:</span>
                                        <span>{(item.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                                <div className="comparison-response">
                                    <strong>Response:</strong>
                                    <p>{item.response}</p>
                                </div>
                            </div>
                        ))}
                    </div>

                    {result.best_response && (
                        <div className="best-response-summary">
                            <h3>✨ Best Response</h3>
                            <p className="best-response-text">{result.best_response.response}</p>
                            <div className="best-response-stats">
                                <span>Score: {result.best_response.score.toFixed(2)}/5.0</span>
                                <span>Quality: {result.best_response.quality}</span>
                                <span>Confidence: {(result.best_response.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default CompareResponses;
