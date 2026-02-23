import { useState } from 'react';
import { batchEvaluate } from '../services/api';

function BatchEvaluator() {
    const [items, setItems] = useState([
        { prompt: '', response: '' },
    ]);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const addItem = () => {
        setItems([...items, { prompt: '', response: '' }]);
    };

    const removeItem = (index) => {
        setItems(items.filter((_, i) => i !== index));
    };

    const updateItem = (index, field, value) => {
        const newItems = [...items];
        newItems[index][field] = value;
        setItems(newItems);
    };

    const handleBatchEvaluate = async () => {
        const validItems = items.filter(item => item.prompt.trim() && item.response.trim());

        if (validItems.length === 0) {
            setError('Please add at least one valid prompt-response pair');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const data = await batchEvaluate(validItems);
            setResults(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to evaluate batch');
        } finally {
            setLoading(false);
        }
    };

    const exportCSV = () => {
        if (!results) return;

        const headers = ['Prompt', 'Response', 'Score', 'Quality', 'Confidence', 'Timestamp'];
        const rows = results.results.map(r => [
            r.prompt,
            r.response,
            r.score,
            r.quality,
            (r.confidence * 100).toFixed(1) + '%',
            r.timestamp
        ]);

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `batch-evaluation-${Date.now()}.csv`;
        link.click();
    };

    return (
        <div className="batch-evaluator">
            <div className="card">
                <div className="card-header">
                    <h2>📦 Batch Evaluation</h2>
                    <p>Evaluate multiple prompt-response pairs at once</p>
                </div>

                <div className="batch-items">
                    {items.map((item, index) => (
                        <div key={index} className="batch-item">
                            <div className="batch-item-header">
                                <span className="batch-item-number">#{index + 1}</span>
                                {items.length > 1 && (
                                    <button
                                        className="btn-remove"
                                        onClick={() => removeItem(index)}
                                        disabled={loading}
                                    >
                                        ✕
                                    </button>
                                )}
                            </div>
                            <div className="batch-item-inputs">
                                <input
                                    type="text"
                                    placeholder="Prompt"
                                    value={item.prompt}
                                    onChange={(e) => updateItem(index, 'prompt', e.target.value)}
                                    disabled={loading}
                                />
                                <input
                                    type="text"
                                    placeholder="Response"
                                    value={item.response}
                                    onChange={(e) => updateItem(index, 'response', e.target.value)}
                                    disabled={loading}
                                />
                            </div>
                        </div>
                    ))}
                </div>

                <div className="button-group">
                    <button
                        className="btn btn-secondary"
                        onClick={addItem}
                        disabled={loading}
                    >
                        + Add Item
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleBatchEvaluate}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <span className="spinner"></span>
                                Evaluating...
                            </>
                        ) : (
                            `Evaluate ${items.filter(i => i.prompt && i.response).length} Items`
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

            {results && (
                <div className="card">
                    <div className="card-header">
                        <h2>📊 Results</h2>
                        <button className="btn btn-secondary" onClick={exportCSV}>
                            💾 Export CSV
                        </button>
                    </div>

                    <div className="summary-stats">
                        <div className="stat-card">
                            <div className="stat-value">{results.summary.total}</div>
                            <div className="stat-label">Total</div>
                        </div>
                        <div className="stat-card success">
                            <div className="stat-value">{results.summary.successful}</div>
                            <div className="stat-label">Successful</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{results.summary.average_score?.toFixed(2)}</div>
                            <div className="stat-label">Avg Score</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{results.summary.min_score?.toFixed(2)}</div>
                            <div className="stat-label">Min</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{results.summary.max_score?.toFixed(2)}</div>
                            <div className="stat-label">Max</div>
                        </div>
                    </div>

                    <div className="results-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Prompt</th>
                                    <th>Response</th>
                                    <th>Score</th>
                                    <th>Quality</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.results.map((result, index) => (
                                    <tr key={index}>
                                        <td>{index + 1}</td>
                                        <td className="text-cell">{result.prompt?.substring(0, 50)}...</td>
                                        <td className="text-cell">{result.response?.substring(0, 50)}...</td>
                                        <td className="score-cell">{result.score?.toFixed(2)}</td>
                                        <td>
                                            <span className={`quality-badge ${result.quality?.toLowerCase()}`}>
                                                {result.quality}
                                            </span>
                                        </td>
                                        <td>{(result.confidence * 100).toFixed(1)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}

export default BatchEvaluator;
