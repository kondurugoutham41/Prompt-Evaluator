"""
FastAPI REST API for Local Prompt Evaluator

Provides endpoints for single evaluation, batch processing, and comparison.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize FastAPI app
from config import API_CONFIG

app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global evaluator instance
evaluator = None


# Pydantic models
class EvaluationRequest(BaseModel):
    prompt: str
    response: str


class EvaluationResponse(BaseModel):
    prompt: str
    response: str
    score: float
    binary_score: float
    quality: str
    confidence: float
    model: str
    timestamp: str


class BatchEvaluationRequest(BaseModel):
    items: List[Dict[str, str]]


class BatchEvaluationResponse(BaseModel):
    results: List[EvaluationResponse]
    summary: Dict[str, Any]


class CompareRequest(BaseModel):
    prompt: str
    responses: List[str]


class CompareResponse(BaseModel):
    prompt: str
    num_responses: int
    ranked_results: List[EvaluationResponse]
    best_response: EvaluationResponse


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    base_model: str
    max_length: int
    score_scale: float
    device: str
    parameters: Dict[str, Any]
    model_path: str
    tokenizer_path: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global evaluator
    
    try:
        from evaluation.evaluator import PromptEvaluator
        evaluator = PromptEvaluator()
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        evaluator = None


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Local Prompt Evaluator API",
        "version": API_CONFIG["version"],
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if evaluator else "model_not_loaded",
        "model_loaded": evaluator is not None,
    }


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_prompt(request: EvaluationRequest):
    """
    Evaluate a single prompt-response pair.
    
    Args:
        request: Prompt and response
    
    Returns:
        Evaluation result with score and quality
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = evaluator.evaluate(
            prompt=request.prompt,
            response=request.response,
        )
        return result
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-evaluate", response_model=BatchEvaluationResponse, tags=["Evaluation"])
async def batch_evaluate(request: BatchEvaluationRequest):
    """
    Evaluate multiple prompt-response pairs.
    
    Args:
        request: List of prompt-response pairs
    
    Returns:
        List of evaluation results with summary statistics
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = evaluator.evaluate_batch(request.items)
        
        # Calculate summary statistics
        scores = [r["score"] for r in results]
        summary = {
            "total": len(results),
            "successful": len(results),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }
        
        return {
            "results": results,
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Batch evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse, tags=["Evaluation"])
async def compare_responses(request: CompareRequest):
    """
    Compare multiple responses to the same prompt.
    
    Args:
        request: Prompt and list of responses
    
    Returns:
        Ranked results with best response highlighted
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = evaluator.compare_responses(
            prompt=request.prompt,
            responses=request.responses,
        )
        return result
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model metadata and configuration."""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return evaluator.get_model_info()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
    )
