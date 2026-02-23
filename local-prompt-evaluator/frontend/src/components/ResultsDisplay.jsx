import ScoreCard from './ScoreCard';

function ResultsDisplay({ result }) {
    const getQualityColor = (quality) => {
        const colors = {
            excellent: '#00E676',
            good: '#00D9FF',
            fair: '#FFB300',
            poor: '#FF5252',
        };
        return colors[quality.toLowerCase()] || '#B0B8C9';
    };

    const copyToClipboard = () => {
        const text = `Score: ${result.score}/5.0\nQuality: ${result.quality}\nConfidence: ${(result.confidence * 100).toFixed(1)}%\nTimestamp: ${result.timestamp}`;
        navigator.clipboard.writeText(text);
        alert('Results copied to clipboard!');
    };

    const exportJSON = () => {
        const dataStr = JSON.stringify(result, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `evaluation-${Date.now()}.json`;
        link.click();
    };

    return (
        <div className="results-display">
            <div className="card">
                <div className="card-header">
                    <h2>📊 Evaluation Results</h2>
                    <div className="action-buttons">
                        <button className="btn-icon" onClick={copyToClipboard} title="Copy to clipboard">
                            📋
                        </button>
                        <button className="btn-icon" onClick={exportJSON} title="Export JSON">
                            💾
                        </button>
                    </div>
                </div>

                <div className="score-section">
                    <ScoreCard
                        score={result.score}
                        quality={result.quality}
                        confidence={result.confidence}
                    />
                </div>

                <div className="details-grid">
                    <div className="detail-item">
                        <span className="detail-label">Binary Score</span>
                        <span className="detail-value">{result.binary_score.toFixed(4)}</span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-label">Confidence</span>
                        <span className="detail-value">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-label">Quality</span>
                        <span
                            className="quality-badge"
                            style={{ backgroundColor: getQualityColor(result.quality) }}
                        >
                            {result.quality.toUpperCase()}
                        </span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-label">Model</span>
                        <span className="detail-value">{result.model}</span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-label">Timestamp</span>
                        <span className="detail-value">{new Date(result.timestamp).toLocaleString()}</span>
                    </div>
                </div>

                <div className="prompt-response-preview">
                    <div className="preview-section">
                        <h3>Prompt</h3>
                        <p className="preview-text">{result.prompt}</p>
                    </div>
                    <div className="preview-section">
                        <h3>Response</h3>
                        <p className="preview-text">{result.response}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ResultsDisplay;
