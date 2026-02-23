function ScoreCard({ score, quality, confidence }) {
    const getQualityColor = (quality) => {
        const colors = {
            excellent: '#00E676',
            good: '#00D9FF',
            fair: '#FFB300',
            poor: '#FF5252',
        };
        return colors[quality.toLowerCase()] || '#B0B8C9';
    };

    const percentage = (score / 5) * 100;
    const color = getQualityColor(quality);

    return (
        <div className="score-card">
            <div className="score-circle-container">
                <svg className="score-circle" viewBox="0 0 200 200">
                    {/* Background circle */}
                    <circle
                        cx="100"
                        cy="100"
                        r="85"
                        fill="none"
                        stroke="#1A1F3A"
                        strokeWidth="15"
                    />
                    {/* Progress circle */}
                    <circle
                        cx="100"
                        cy="100"
                        r="85"
                        fill="none"
                        stroke={color}
                        strokeWidth="15"
                        strokeLinecap="round"
                        strokeDasharray={`${percentage * 5.34} 534`}
                        transform="rotate(-90 100 100)"
                        className="score-progress"
                    />
                </svg>
                <div className="score-text">
                    <div className="score-value">{score.toFixed(2)}</div>
                    <div className="score-max">/5.0</div>
                </div>
            </div>

            <div className="score-info">
                <div className="quality-label" style={{ color }}>
                    {quality.toUpperCase()}
                </div>
                <div className="confidence-bar">
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-track">
                        <div
                            className="confidence-fill"
                            style={{
                                width: `${confidence * 100}%`,
                                backgroundColor: color
                            }}
                        ></div>
                    </div>
                    <div className="confidence-value">{(confidence * 100).toFixed(1)}%</div>
                </div>
            </div>
        </div>
    );
}

export default ScoreCard;
