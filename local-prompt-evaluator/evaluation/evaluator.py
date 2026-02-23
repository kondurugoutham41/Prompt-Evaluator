"""
Prompt Evaluator - Inference Engine

Loads trained model and evaluates prompts locally.
"""

import torch
import pandas as pd
from transformers import DistilBertTokenizer
from pathlib import Path
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class PromptEvaluator:
    """Local prompt evaluator using fine-tuned DistilBERT."""
    
    def __init__(self, model_path: Path = None, tokenizer_path: Path = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model (None = use config)
            tokenizer_path: Path to tokenizer (None = use config)
        """
        from config import MODEL_CONFIG, EVALUATION_CONFIG
        from training.model import PromptEvaluatorModel
        
        self.model_config = MODEL_CONFIG
        self.eval_config = EVALUATION_CONFIG
        
        model_path = model_path or MODEL_CONFIG["model_path"]
        tokenizer_path = tokenizer_path or MODEL_CONFIG["tokenizer_path"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model from {model_path}")
        self.model = PromptEvaluatorModel.from_pretrained(model_path, self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"Model loaded on {self.device}")
    
    def evaluate(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a single prompt-response pair.
        
        Args:
            prompt: User's prompt
            response: AI's response
        
        Returns:
            Dictionary with score, quality, confidence, etc.
        """
        # Format input
        text = f"Prompt: {prompt}\n\nResponse: {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.model_config["max_length"],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            binary_score = self.model(input_ids, attention_mask).item()
        
        # Scale to 0-5
        score = binary_score * self.eval_config["score_scale"]
        
        # Determine quality
        quality = self._get_quality_label(binary_score)
        
        # Calculate confidence
        confidence = abs(binary_score - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
        
        return {
            "prompt": prompt,
            "response": response,
            "score": score,
            "binary_score": binary_score,
            "quality": quality,
            "confidence": confidence,
            "model": self.model_config["base_model"],
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    
    def evaluate_batch(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompt-response pairs.
        
        Args:
            items: List of dicts with 'prompt' and 'response' keys
        
        Returns:
            List of evaluation results
        """
        results = []
        for item in items:
            result = self.evaluate(item["prompt"], item["response"])
            results.append(result)
        return results
    
    def compare_responses(
        self,
        prompt: str,
        responses: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple responses to the same prompt.
        
        Args:
            prompt: User's prompt
            responses: List of AI responses
        
        Returns:
            Dictionary with ranked results
        """
        results = []
        for response in responses:
            result = self.evaluate(prompt, response)
            results.append(result)
        
        # Sort by score (descending)
        ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return {
            "prompt": prompt,
            "num_responses": len(responses),
            "ranked_results": ranked_results,
            "best_response": ranked_results[0],
            "worst_response": ranked_results[-1],
        }
    
    def _get_quality_label(self, binary_score: float) -> str:
        """Get quality label from binary score."""
        thresholds = self.eval_config["quality_thresholds"]
        
        if binary_score >= thresholds["excellent"]:
            return "excellent"
        elif binary_score >= thresholds["good"]:
            return "good"
        elif binary_score >= thresholds["fair"]:
            return "fair"
        else:
            return "poor"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "base_model": self.model_config["base_model"],
            "max_length": self.model_config["max_length"],
            "score_scale": self.eval_config["score_scale"],
            "device": str(self.device),
            "parameters": {
                "distilbert": self.model.distilbert.num_parameters(),
                "classifier": self.model.classifier.out_features,
            },
            "model_path": str(self.model_config["model_path"]),
            "tokenizer_path": str(self.model_config["tokenizer_path"]),
        }



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    evaluator = PromptEvaluator()
    
    result = evaluator.evaluate(
        prompt="What is machine learning?",
        response="Machine learning is a subset of AI that enables systems to learn from data."
    )
    
    print(f"\nScore: {result['score']:.2f}/5.0")
    print(f"Quality: {result['quality']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
