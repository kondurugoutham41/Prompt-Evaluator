"""
DistilBERT-based Prompt Evaluator Model

Architecture:
- Base: distilbert-base-uncased (66M parameters)
- Head: Linear(768 → 1) + Sigmoid
- Output: Binary score [0, 1] representing quality
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PromptEvaluatorModel(nn.Module):
    """
    DistilBERT-based model for evaluating prompt-response quality.
    
    Architecture:
        Input: Tokenized prompt + response (max 512 tokens)
        ↓
        DistilBERT Encoder (6 layers, 768 hidden)
        ↓
        [CLS] token representation
        ↓
        Dropout (0.1)
        ↓
        Linear(768 → 1)
        ↓
        Sigmoid
        ↓
        Output: Score ∈ [0, 1]
    """
    
    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
    ):
        """
        Initialize the model.
        
        Args:
            base_model_name: Hugging Face model identifier
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(base_model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"Initialized model with {base_model_name}")
        logger.info(f"Total parameters: {self.count_parameters():,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            scores: Quality scores [batch_size] ∈ [0, 1]
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Apply dropout and classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, 1]
        scores = self.sigmoid(logits).squeeze(-1)  # [batch_size]
        
        return scores
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_directory: Path):
        """
        Save model weights and config.
        
        Args:
            save_directory: Directory to save model
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save DistilBERT
        self.distilbert.save_pretrained(save_directory)
        
        # Save classifier head
        classifier_path = save_directory / "classifier.pt"
        torch.save({
            'classifier': self.classifier.state_dict(),
            'dropout': self.dropout.p,
        }, classifier_path)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_directory: Path,
        device: Optional[torch.device] = None,
    ) -> "PromptEvaluatorModel":
        """
        Load model from saved weights.
        
        Args:
            model_directory: Directory containing saved model
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        model_directory = Path(model_directory)
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DistilBERT
        distilbert = DistilBertModel.from_pretrained(model_directory)
        
        # Create model instance
        model = cls(base_model_name=model_directory)
        model.distilbert = distilbert
        
        # Load classifier head
        classifier_path = model_directory / "classifier.pt"
        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=device)
            model.classifier.load_state_dict(checkpoint['classifier'])
            model.dropout.p = checkpoint['dropout']
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {model_directory}")
        return model


def create_model(config: dict) -> PromptEvaluatorModel:
    """
    Create model from config.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    """
    return PromptEvaluatorModel(
        base_model_name=config.get("base_model", "distilbert-base-uncased"),
        dropout=config.get("dropout", 0.1),
    )
