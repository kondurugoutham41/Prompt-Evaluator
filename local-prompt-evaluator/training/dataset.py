"""
PyTorch Dataset for Prompt-Response Pairs

Loads CSV data and tokenizes for DistilBERT input.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class PromptResponseDataset(Dataset):
    """
    Dataset for prompt-response pairs with binary labels.
    
    Format:
        CSV with columns: prompt, response, label
        - prompt: User's question/instruction
        - response: AI's response
        - label: 0 (poor) or 1 (good)
    """
    
    def __init__(
        self,
        csv_path: Path,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        from config import DATASET_CONFIG
        max_samples = DATASET_CONFIG.get("max_samples", None)
        self.df = pd.read_csv(csv_path, nrows=max_samples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loaded dataset: {csv_path}")
        logger.info(f"  Samples: {len(self.df)}")
        logger.info(f"  Positive: {self.df['label'].sum()}")
        logger.info(f"  Negative: {len(self.df) - self.df['label'].sum()}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        row = self.df.iloc[idx]
        
        # Use pre-combined text column (format: [PROMPT] ... [RESPONSE] ...)
        text = str(row['text'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.float),
        }


def create_dataloaders(
    train_path: Path,
    test_path: Path,
    tokenizer: DistilBertTokenizer,
    batch_size: int,
    max_length: int = 512,
    num_workers: int = 0,
):
    """
    Create train and test dataloaders.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        tokenizer: DistilBERT tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = PromptResponseDataset(train_path, tokenizer, max_length)
    test_dataset = PromptResponseDataset(test_path, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader
