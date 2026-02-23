"""
Example Usage of Local Prompt Evaluator

Demonstrates how to use the evaluator programmatically.
"""

from evaluation.evaluator import PromptEvaluator
import logging

logging.basicConfig(level=logging.INFO)


def example_single_evaluation():
    """Example: Single prompt evaluation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Evaluation")
    print("=" * 80)
    
    evaluator = PromptEvaluator()
    
    result = evaluator.evaluate(
        prompt="What is machine learning?",
        response="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    )
    
    print(f"\nPrompt: {result['prompt']}")
    print(f"Response: {result['response'][:100]}...")
    print(f"\nScore: {result['score']:.2f}/5.0")
    print(f"Quality: {result['quality'].upper()}")
    print(f"Confidence: {result['confidence']*100:.1f}%")


def example_batch_evaluation():
    """Example: Batch evaluation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Evaluation")
    print("=" * 80)
    
    evaluator = PromptEvaluator()
    
    items = [
        {
            "prompt": "Explain neural networks",
            "response": "Neural networks are computing systems inspired by biological neural networks."
        },
        {
            "prompt": "What is deep learning?",
            "response": "Deep learning uses multiple layers to progressively extract higher-level features."
        },
        {
            "prompt": "Define AI",
            "response": "AI is intelligence demonstrated by machines."
        },
    ]
    
    results = evaluator.evaluate_batch(items)
    
    print(f"\nEvaluated {len(results)} items:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.2f}/5.0 | Quality: {result['quality']}")
        print(f"   Prompt: {result['prompt'][:50]}...")


def example_compare_responses():
    """Example: Compare multiple responses."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Responses")
    print("=" * 80)
    
    evaluator = PromptEvaluator()
    
    prompt = "What is the capital of France?"
    responses = [
        "Paris is the capital of France.",
        "The capital of France is Paris, a major European city.",
        "Paris.",
    ]
    
    result = evaluator.compare_responses(prompt, responses)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nRanked Results:")
    for i, res in enumerate(result['ranked_results'], 1):
        print(f"\n{i}. Score: {res['score']:.2f}/5.0")
        print(f"   Response: {res['response']}")


if __name__ == "__main__":
    print("\nLocal Prompt Evaluator - Examples\n")
    
    try:
        example_single_evaluation()
        example_batch_evaluation()
        example_compare_responses()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you've trained the model first:")
        print("  python main.py full")
