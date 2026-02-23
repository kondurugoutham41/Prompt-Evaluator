"""
Local Prompt Evaluator - Main CLI Interface

Unified command-line interface for all operations:
- prepare: Download and prepare HelpSteer2 dataset
- train: Train the DistilBERT model
- evaluate: Interactive prompt evaluation
- api: Start REST API server
- test: Run verification tests
- full: Complete pipeline (prepare + train)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOGGING_CONFIG,
    TRAINING_CONFIG,
    API_CONFIG,
    MODEL_CONFIG,
    DATASET_CONFIG,
)

# Setup logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def command_prepare(args):
    """Prepare dataset from HelpSteer2."""
    from training.prepare_data import prepare_data
    
    logger.info("=" * 80)
    logger.info("PREPARING DATASET")
    logger.info("=" * 80)
    
    try:
        train_df, test_df = prepare_data(max_samples=args.max_samples)
        logger.info(f"✓ Dataset prepared successfully")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def command_train(args):
    """Train the model."""
    from training.train import train_model
    
    logger.info("=" * 80)
    logger.info("TRAINING MODEL")
    logger.info("=" * 80)
    
    try:
        train_model()
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def command_evaluate(args):
    """Interactive evaluation."""
    from evaluation.evaluator import PromptEvaluator
    
    logger.info("=" * 80)
    logger.info("INTERACTIVE EVALUATION")
    logger.info("=" * 80)
    
    try:
        evaluator = PromptEvaluator()
        
        print("\nEnter 'quit' to exit\n")
        
        while True:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            response = input("Response: ").strip()
            if response.lower() == 'quit':
                break
            
            result = evaluator.evaluate(prompt, response)
            
            print(f"\n{'='*60}")
            print(f"Score: {result['score']:.2f}/5.0")
            print(f"Quality: {result['quality'].upper()}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"{'='*60}\n")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def command_api(args):
    """Start API server."""
    import uvicorn
    
    logger.info("=" * 80)
    logger.info("STARTING API SERVER")
    logger.info("=" * 80)
    
    try:
        uvicorn.run(
            "api.app:app",
            host=API_CONFIG["host"],
            port=API_CONFIG["port"],
            reload=API_CONFIG["reload"],
            workers=API_CONFIG["workers"],
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def command_test(args):
    """Run verification tests."""
    from evaluation.evaluator import PromptEvaluator
    
    logger.info("=" * 80)
    logger.info("RUNNING TESTS")
    logger.info("=" * 80)
    
    try:
        # Test 1: Load model
        print("\n[1/3] Testing model loading...")
        evaluator = PromptEvaluator()
        print("✓ Model loaded successfully")
        
        # Test 2: Single evaluation
        print("\n[2/3] Testing single evaluation...")
        result = evaluator.evaluate(
            prompt="What is AI?",
            response="AI is artificial intelligence."
        )
        print(f"✓ Evaluation successful: {result['score']:.2f}/5.0")
        
        # Test 3: Batch evaluation
        print("\n[3/3] Testing batch evaluation...")
        results = evaluator.evaluate_batch([
            {"prompt": "Test 1", "response": "Response 1"},
            {"prompt": "Test 2", "response": "Response 2"},
        ])
        print(f"✓ Batch evaluation successful: {len(results)} results")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def command_full(args):
    """Run full pipeline."""
    logger.info("=" * 80)
    logger.info("FULL PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Prepare data
    logger.info("\n[1/3] Preparing data...")
    command_prepare(args)
    
    # Step 2: Train model
    logger.info("\n[2/3] Training model...")
    command_train(args)
    
    # Step 3: Test
    logger.info("\n[3/3] Testing...")
    command_test(args)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE ✓")
    logger.info("=" * 80)


def command_info(args):
    """Show configuration info."""
    print("\n" + "=" * 80)
    print("LOCAL PROMPT EVALUATOR - CONFIGURATION")
    print("=" * 80)
    
    print(f"\nDataset:")
    print(f"  Name: {DATASET_CONFIG['name']}")
    print(f"  Train file: {DATASET_CONFIG['train_file']}")
    print(f"  Test file: {DATASET_CONFIG['test_file']}")
    
    print(f"\nModel:")
    print(f"  Base: {MODEL_CONFIG['base_model']}")
    print(f"  Max length: {MODEL_CONFIG['max_length']}")
    print(f"  Path: {MODEL_CONFIG['model_path']}")
    
    print(f"\nTraining:")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Device: {TRAINING_CONFIG['device']}")
    
    print(f"\nAPI:")
    print(f"  Host: {API_CONFIG['host']}")
    print(f"  Port: {API_CONFIG['port']}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Local Prompt Evaluator - CLI Interface"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # prepare
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--max-samples", type=int, help="Max samples to use")
    
    # train
    train_parser = subparsers.add_parser("train", help="Train model")
    
    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Interactive evaluation")
    
    # api
    api_parser = subparsers.add_parser("api", help="Start API server")
    
    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    
    # full
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--max-samples", type=int, help="Max samples to use")
    
    # info
    info_parser = subparsers.add_parser("info", help="Show configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "prepare": command_prepare,
        "train": command_train,
        "evaluate": command_evaluate,
        "api": command_api,
        "test": command_test,
        "full": command_full,
        "info": command_info,
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
