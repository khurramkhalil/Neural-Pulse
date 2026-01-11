#!/usr/bin/env python3
"""
Debug script to identify source of NaN entropy values.

This script tests the entropy computation on a simple example
and adds detailed logging to trace where NaN originates.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def compute_entropy_debug(logits: torch.Tensor) -> float:
    """
    Debug version of compute_entropy with extensive logging.
    """
    logger.info("="*80)
    logger.info("ENTROPY COMPUTATION DEBUG")
    logger.info("="*80)

    logger.info(f"Input logits shape: {logits.shape}")
    logger.info(f"Input logits dtype: {logits.dtype}")
    logger.info(f"Input logits device: {logits.device}")
    logger.info(f"Logits min: {logits.min().item():.4f}")
    logger.info(f"Logits max: {logits.max().item():.4f}")
    logger.info(f"Logits has NaN: {torch.isnan(logits).any().item()}")
    logger.info(f"Logits has Inf: {torch.isinf(logits).any().item()}")

    # Step 1: Softmax
    logger.info("\nStep 1: Computing softmax...")
    probs = torch.softmax(logits, dim=-1)
    logger.info(f"Probs shape: {probs.shape}")
    logger.info(f"Probs dtype: {probs.dtype}")
    logger.info(f"Probs min: {probs.min().item():.10f}")
    logger.info(f"Probs max: {probs.max().item():.10f}")
    logger.info(f"Probs sum: {probs.sum().item():.10f}")
    logger.info(f"Probs has NaN: {torch.isnan(probs).any().item()}")
    logger.info(f"Probs has Inf: {torch.isinf(probs).any().item()}")
    logger.info(f"Number of zeros in probs: {(probs == 0).sum().item()}")
    logger.info(f"Number of very small probs (<1e-10): {(probs < 1e-10).sum().item()}")

    # Step 2: Add epsilon
    logger.info("\nStep 2: Adding epsilon (1e-10)...")
    probs_eps = probs + 1e-10
    logger.info(f"Probs+eps min: {probs_eps.min().item():.10f}")
    logger.info(f"Probs+eps max: {probs_eps.max().item():.10f}")

    # Step 3: Log
    logger.info("\nStep 3: Computing log...")
    log_probs = torch.log(probs_eps)
    logger.info(f"Log_probs shape: {log_probs.shape}")
    logger.info(f"Log_probs dtype: {log_probs.dtype}")
    logger.info(f"Log_probs min: {log_probs.min().item():.4f}")
    logger.info(f"Log_probs max: {log_probs.max().item():.4f}")
    logger.info(f"Log_probs has NaN: {torch.isnan(log_probs).any().item()}")
    logger.info(f"Log_probs has Inf: {torch.isinf(log_probs).any().item()}")
    logger.info(f"Number of -Inf in log_probs: {torch.isinf(log_probs).sum().item()}")

    # Step 4: Multiply
    logger.info("\nStep 4: Computing probs * log_probs...")
    product = probs * log_probs
    logger.info(f"Product shape: {product.shape}")
    logger.info(f"Product dtype: {product.dtype}")
    logger.info(f"Product min: {product.min().item():.4f}")
    logger.info(f"Product max: {product.max().item():.4f}")
    logger.info(f"Product has NaN: {torch.isnan(product).any().item()}")
    logger.info(f"Product has Inf: {torch.isinf(product).any().item()}")

    # Step 5: Sum
    logger.info("\nStep 5: Computing sum...")
    sum_val = torch.sum(product)
    logger.info(f"Sum value: {sum_val.item():.4f}")
    logger.info(f"Sum has NaN: {torch.isnan(sum_val).item()}")
    logger.info(f"Sum has Inf: {torch.isinf(sum_val).item()}")

    # Step 6: Negate
    logger.info("\nStep 6: Negating...")
    entropy = -sum_val.item()
    logger.info(f"Final entropy: {entropy:.4f}")
    logger.info(f"Entropy is NaN: {entropy != entropy}")

    logger.info("="*80)

    return entropy


def test_simple_case():
    """Test on a simple 2+2 example"""
    logger.info("\n" + "="*80)
    logger.info("TESTING SIMPLE CASE: 'What is 2+2?'")
    logger.info("="*80 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    logger.info(f"Device: {device}")
    logger.info(f"Dtype: {dtype}\n")

    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    logger.info(f"Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map='auto' if device == 'cuda' else None,
        output_attentions=True
    )

    if device == 'cpu':
        model.to(device)

    model.eval()
    logger.info("Model loaded successfully\n")

    # Prepare prompt
    prompt = "What is 2+2?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    logger.info(f"Formatted prompt: {formatted_prompt[:100]}...\n")

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(device)
    logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
    logger.info(f"Input length: {inputs['input_ids'].size(1)} tokens\n")

    # Generate first token
    logger.info("Generating first token...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits[0, -1, :]  # Last token logits

        logger.info(f"\nExtracted logits for first generation step")

        # Test entropy computation
        entropy = compute_entropy_debug(logits)

        logger.info(f"\n{'='*80}")
        logger.info(f"RESULT: Entropy = {entropy}")
        logger.info(f"Is NaN: {entropy != entropy}")
        logger.info(f"{'='*80}\n")


def test_synthetic_data():
    """Test on synthetic logits to isolate the issue"""
    logger.info("\n" + "="*80)
    logger.info("TESTING SYNTHETIC DATA")
    logger.info("="*80 + "\n")

    # Test 1: Uniform distribution
    logger.info("Test 1: Uniform distribution (should have high entropy)")
    logits = torch.ones(1000, dtype=torch.float32)
    entropy = compute_entropy_debug(logits)
    logger.info(f"Result: {entropy:.4f} (expected ~6.91)")

    # Test 2: One-hot distribution
    logger.info("\n\nTest 2: One-hot distribution (should have low entropy)")
    logits = torch.zeros(1000, dtype=torch.float32)
    logits[42] = 100.0
    entropy = compute_entropy_debug(logits)
    logger.info(f"Result: {entropy:.4f} (expected ~0.0)")

    # Test 3: Half precision (like CUDA)
    logger.info("\n\nTest 3: Half precision (like CUDA would use)")
    logits = torch.randn(1000, dtype=torch.float16)
    entropy = compute_entropy_debug(logits)
    logger.info(f"Result: {entropy:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Debug entropy computation')
    parser.add_argument('--mode', choices=['simple', 'synthetic', 'both'],
                       default='both',
                       help='Test mode')

    args = parser.parse_args()

    if args.mode in ['synthetic', 'both']:
        test_synthetic_data()

    if args.mode in ['simple', 'both']:
        test_simple_case()


if __name__ == '__main__':
    main()
