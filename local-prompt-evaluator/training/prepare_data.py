"""
Data Preparation Pipeline for HelpSteer2 Dataset

Loads nvidia/HelpSteer2, converts to binary classification format,
and saves as CSV files for training.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def load_helpsteer2_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load HelpSteer2 dataset from Hugging Face.
    
    Args:
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        DataFrame with prompt, response, and helpfulness scores
    """
    logger.info("Loading HelpSteer2 dataset from Hugging Face...")
    
    dataset = load_dataset("nvidia/HelpSteer2", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    df = pd.DataFrame(dataset)
    
    logger.info(f"Loaded {len(df)} samples")
    return df


def convert_to_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert HelpSteer2 to binary classification format.
    
    Uses helpfulness score as quality indicator:
    - helpfulness >= 2.5 → label = 1 (good)
    - helpfulness < 2.5 → label = 0 (poor)
    
    Args:
        df: Raw HelpSteer2 DataFrame
    
    Returns:
        Processed DataFrame with prompt, response, label
    """
    logger.info("Converting to binary classification format...")
    
    processed = pd.DataFrame({
        'prompt': df['prompt'],
        'response': df['response'],
        'label': (df['helpfulness'] >= 2.5).astype(int)
    })
    
    # Remove any rows with missing values
    processed = processed.dropna()
    
    logger.info(f"Processed {len(processed)} samples")
    logger.info(f"  Positive (good): {processed['label'].sum()}")
    logger.info(f"  Negative (poor): {len(processed) - processed['label'].sum()}")
    
    return processed


def split_and_save_dataset(
    df: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    split_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset and save to CSV files.
    
    Args:
        df: Processed DataFrame
        train_path: Path to save training data
        test_path: Path to save test data
        split_ratio: Train/test split ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting dataset ({split_ratio:.0%} train, {1-split_ratio:.0%} test)...")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Save to CSV
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved training data: {train_path} ({len(train_df)} samples)")
    logger.info(f"Saved test data: {test_path} ({len(test_df)} samples)")
    
    return train_df, test_df


def prepare_data(max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline.
    
    Args:
        max_samples: Maximum samples to use (None = all)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    from config import DATASET_CONFIG
    
    logger.info("=" * 80)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 80)
    
    # Load dataset
    raw_df = load_helpsteer2_dataset(max_samples)
    
    # Convert to binary classification
    processed_df = convert_to_binary_classification(raw_df)
    
    # Split and save
    train_df, test_df = split_and_save_dataset(
        processed_df,
        DATASET_CONFIG["train_file"],
        DATASET_CONFIG["test_file"],
        DATASET_CONFIG["split_ratio"],
        DATASET_CONFIG["random_seed"],
    )
    
    logger.info("=" * 80)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data()
