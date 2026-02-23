import { useState } from 'react';
import { evaluatePrompt } from '../services/api';
import ResultsDisplay from './ResultsDisplay';

function PromptInput() {
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleEvaluate = async () => {
        if (!prompt.trim() || !response.trim()) {
            setError('Please enter both prompt and response');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const data = await evaluatePrompt(prompt, response);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to evaluate. Make sure the API is running.');
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setPrompt('');
        setResponse('');
        setResult(null);
        setError(null);
    };

    return (
        <div className="prompt-input-container">
            <div className="input-section">
                <div className="card">
                    <div className="card-header">
                        <h2>📝 Evaluate Prompt & Response</h2>
                        <p>Enter your prompt and AI response to get quality scores</p>
                    </div>

                    <div className="form-group">
                        <label htmlFor="prompt">
                            Prompt
                            <span className="char-count">{prompt.length} characters</span>
                        </label>
                        <textarea
                            id="prompt"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Enter your prompt here... (e.g., 'Explain quantum computing in simple terms')"
                            rows={4}
                            disabled={loading}
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="response">
                            AI Response
                            <span className="char-count">{response.length} characters</span>
                        </label>
                        <textarea
                            id="response"
                            value={response}
                            onChange={(e) => setResponse(e.target.value)}
                            placeholder="Enter the AI-generated response here..."
                            rows={6}
                            disabled={loading}
                        />
                    </div>

                    {error && (
                        <div className="error-message">
                            <span className="error-icon">⚠️</span>
                            {error}
                        </div>
                    )}

                    <div className="button-group">
                        <button
                            className="btn btn-primary"
                            onClick={handleEvaluate}
                            disabled={loading || !prompt.trim() || !response.trim()}
                        >
                            {loading ? (
                                <>
                                    <span className="spinner"></span>
                                    Evaluating...
                                </>
                            ) : (
                                <>
                                    <span>✨</span>
                                    Evaluate
                                </>
                            )}
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={handleClear}
                            disabled={loading}
                        >
                            Clear
                        </button>
                    </div>
                </div>
            </div>

            {result && (
                <div className="results-section">
                    <ResultsDisplay result={result} />
                </div>
            )}
        </div>
    );
}

export default PromptInput;
