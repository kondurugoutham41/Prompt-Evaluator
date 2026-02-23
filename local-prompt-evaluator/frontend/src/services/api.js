/**
 * API Service Layer for Local Prompt Evaluator
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const evaluatePrompt = async (prompt, response) => {
    const res = await api.post('/evaluate', { prompt, response });
    return res.data;
};

export const batchEvaluate = async (items) => {
    const res = await api.post('/batch-evaluate', { items });
    return res.data;
};

export const compareResponses = async (prompt, responses) => {
    const res = await api.post('/compare', { prompt, responses });
    return res.data;
};

export const getHealth = async () => {
    const res = await api.get('/health');
    return res.data;
};

export const getModelInfo = async () => {
    const res = await api.get('/model-info');
    return res.data;
};

export default {
    evaluatePrompt,
    batchEvaluate,
    compareResponses,
    getHealth,
    getModelInfo,
};
