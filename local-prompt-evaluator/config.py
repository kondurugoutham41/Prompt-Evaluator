"""
Central Configuration for Local Prompt Evaluator

All hyperparameters, paths, and settings in one place.
Override via environment variables for production.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    "name": "nvidia/HelpSteer2",
    "split": "train",
    "train_file": DATA_DIR / "train.csv",
    "test_file": DATA_DIR / "test.csv",
    "split_ratio": 0.8,  # 80% train, 20% test
    "random_seed": 42,
    "max_samples": 500,  # Limit samples for quick test run (None = use all)
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "base_model": "distilbert-base-uncased",
    "max_length": 512,  # Maximum sequence length
    "dropout": 0.1,  # Dropout probability
    "model_path": MODELS_DIR / "prompt_evaluator",
    "tokenizer_path": MODELS_DIR / "tokenizer",
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Data
    "batch_size": int(os.getenv("BATCH_SIZE", "8")),  # Smaller batch for quick test
    "num_workers": 0,  # DataLoader workers (0 for Windows compatibility)
    
    # Training
    "epochs": int(os.getenv("EPOCHS", "1")),
    "learning_rate": float(os.getenv("LEARNING_RATE", "2e-5")),
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    
    # Device
    "device": os.getenv("DEVICE", "cpu"),  # "cuda" or "cpu"
    
    # Logging
    "log_interval": 50,  # Log every N batches
    "save_interval": 1,  # Save every N epochs
    "log_file": LOGS_DIR / "training_history.csv",
    
    # Early stopping
    "patience": 3,  # Stop if no improvement for N epochs
    "min_delta": 0.001,  # Minimum improvement threshold
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    "score_scale": 5.0,  # Scale binary [0,1] to [0,5]
    "quality_thresholds": {
        "excellent": 0.8,  # >= 0.8 → 4.0-5.0
        "good": 0.6,       # >= 0.6 → 3.0-3.99
        "fair": 0.4,       # >= 0.4 → 2.0-2.99
        "poor": 0.0,       # < 0.4  → 0.0-1.99
    },
}


# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("API_PORT", "8000")),
    "reload": True,  # Auto-reload on code changes (dev only)
    "workers": 1,
    "cors_origins": ["*"],  # Allow all origins (restrict in production)
    "title": "Local Prompt Evaluator API",
    "description": "REST API for evaluating prompts using fine-tuned DistilBERT",
    "version": "1.0.0",
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "app.log",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

PROMPT_TEMPLATE = """Prompt: {prompt}

Response: {response}"""


def format_prompt_response(prompt: str, response: str) -> str:
    """Format prompt and response for model input."""
    return PROMPT_TEMPLATE.format(prompt=prompt, response=response)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    assert TRAINING_CONFIG["batch_size"] > 0, "Batch size must be positive"
    assert TRAINING_CONFIG["epochs"] > 0, "Epochs must be positive"
    assert 0 < DATASET_CONFIG["split_ratio"] < 1, "Split ratio must be between 0 and 1"
    assert MODEL_CONFIG["max_length"] > 0, "Max length must be positive"
    print("✓ Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print("\nConfiguration Summary:")
    print(f"  Dataset: {DATASET_CONFIG['name']}")
    print(f"  Model: {MODEL_CONFIG['base_model']}")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Device: {TRAINING_CONFIG['device']}")
    print(f"  API Port: {API_CONFIG['port']}")
