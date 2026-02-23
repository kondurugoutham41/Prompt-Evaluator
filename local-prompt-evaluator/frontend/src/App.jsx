import { useState, useEffect } from 'react';
import PromptInput from './components/PromptInput';
import BatchEvaluator from './components/BatchEvaluator';
import CompareResponses from './components/CompareResponses';
import { getHealth, getModelInfo } from './services/api';

function App() {
    const [activeTab, setActiveTab] = useState('single');
    const [modelInfo, setModelInfo] = useState(null);
    const [apiStatus, setApiStatus] = useState('checking');

    // Check API health on mount
    useEffect(() => {
        checkHealth();
        fetchModelInfo();
    }, []);

    const checkHealth = async () => {
        try {
            await getHealth();
            setApiStatus('healthy');
        } catch (error) {
            setApiStatus('offline');
        }
    };

    const fetchModelInfo = async () => {
        try {
            const info = await getModelInfo();
            setModelInfo(info);
        } catch (error) {
            console.error('Failed to fetch model info:', error);
        }
    };

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="header-content">
                    <div className="logo-section">
                        <div className="logo-icon">🤖</div>
                        <div>
                            <h1 className="title">Local Prompt Evaluator</h1>
                            <p className="subtitle">Fine-tuned DistilBERT • 100% Local • Zero API Costs</p>
                        </div>
                    </div>
                    <div className="status-section">
                        <div className={`status-indicator ${apiStatus}`}>
                            <span className="status-dot"></span>
                            <span className="status-text">
                                {apiStatus === 'healthy' ? 'API Online' :
                                    apiStatus === 'offline' ? 'API Offline' : 'Checking...'}
                            </span>
                        </div>
                        {modelInfo && (
                            <div className="model-badge">
                                {modelInfo.base_model}
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <nav className="tabs">
                <button
                    className={`tab ${activeTab === 'single' ? 'active' : ''}`}
                    onClick={() => setActiveTab('single')}
                >
                    <span className="tab-icon">📝</span>
                    Single Evaluation
                </button>
                <button
                    className={`tab ${activeTab === 'batch' ? 'active' : ''}`}
                    onClick={() => setActiveTab('batch')}
                >
                    <span className="tab-icon">📦</span>
                    Batch Processing
                </button>
                <button
                    className={`tab ${activeTab === 'compare' ? 'active' : ''}`}
                    onClick={() => setActiveTab('compare')}
                >
                    <span className="tab-icon">🔄</span>
                    Compare Responses
                </button>
            </nav>

            {/* Main Content */}
            <main className="main-content">
                {activeTab === 'single' && <PromptInput />}
                {activeTab === 'batch' && <BatchEvaluator />}
                {activeTab === 'compare' && <CompareResponses />}
            </main>

            {/* Footer */}
            <footer className="footer">
                <p>
                    Built with ❤️ using PyTorch, Transformers, and React •
                    <a href="https://github.com" target="_blank" rel="noopener noreferrer"> View on GitHub</a>
                </p>
            </footer>
        </div>
    );
}

export default App;
