#!/usr/bin/env python3
"""
Generate Traces for SECA Attacks (Batch Processing)

Generates entropy and attention traces for all attacks in a dataset.
This is needed for Phase 2 analysis to identify the "waffling signature".

Usage:
    python scripts/generate_traces_batch.py \\
        --attacks seca_attacks_pilot_100.json \\
        --output datasets/pilot_traces.json \\
        --validation datasets/pilot_validation.json \\
        --model meta-llama/Llama-3.1-8B-Instruct
"""

import json
import argparse
import logging
from typing import List, Dict
from tqdm import tqdm
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.llama_hook import LlamaSignalHook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_traces_batch(
    attacks_path: str,
    output_traces_path: str,
    output_validation_path: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens: int = 100
):
    """
    Generate traces for all attacks in dataset.

    Args:
        attacks_path: Path to attacks JSON
        output_traces_path: Path to save traces
        output_validation_path: Path to save validation labels
        model_name: Model to use for generation
        max_new_tokens: Max tokens to generate per prompt
    """
    # Load attacks
    logger.info(f"Loading attacks from {attacks_path}")
    with open(attacks_path) as f:
        data = json.load(f)

    attacks = data['attacks']
    logger.info(f"Loaded {len(attacks)} attacks")

    # Initialize hook
    logger.info(f"Initializing LlamaSignalHook with {model_name}")
    hook = LlamaSignalHook(model_name=model_name)

    # Generate traces
    traces = []
    validations = []

    logger.info("Generating traces...")
    for i, attack in enumerate(tqdm(attacks, desc="Processing attacks")):
        try:
            # Generate trace for original prompt
            logger.debug(f"Processing attack {i+1}/{len(attacks)}: Original prompt")
            original_trace = hook.generate_with_signals(
                prompt=attack['original_prompt'],
                max_new_tokens=max_new_tokens
            )

            # Generate trace for adversarial prompt
            logger.debug(f"Processing attack {i+1}/{len(attacks)}: Adversarial prompt")
            adversarial_trace = hook.generate_with_signals(
                prompt=attack['adversarial_prompt'],
                max_new_tokens=max_new_tokens
            )

            # Create trace objects
            # Original prompt (non-hallucination - control)
            traces.append({
                'prompt': attack['original_prompt'],
                'generated_text': original_trace.generated_text,
                'entropy_trace': original_trace.entropy_trace,
                'attention_trace': original_trace.attention_trace,  # DEPRECATED (Phase 2 - failed)
                'perplexity_trace': original_trace.perplexity_trace,  # DEPRECATED (Phase 2 - wrong sign)
                'attention_entropy_trace': original_trace.attention_entropy_trace,  # DEPRECATED (Phase 2 - weak)
                'semantic_drift_trace': original_trace.semantic_drift_trace,  # PHASE 2a - PRIMARY SIGNAL
                'attack_id': i,
                'is_adversarial': False
            })

            validations.append({
                'is_hallucination': False,  # Original prompts are NOT hallucinations
                'correctness_score': 1.0,    # Assume original is correct
                'adversarial_score': 0.0,    # Original has no adversarial score
                'equivalence_score': 1.0,    # Original is identical to itself
                'attack_id': i
            })

            # Adversarial prompt (potential hallucination)
            traces.append({
                'prompt': attack['adversarial_prompt'],
                'generated_text': adversarial_trace.generated_text,
                'entropy_trace': adversarial_trace.entropy_trace,
                'attention_trace': adversarial_trace.attention_trace,  # DEPRECATED (Phase 2 - failed)
                'perplexity_trace': adversarial_trace.perplexity_trace,  # DEPRECATED (Phase 2 - wrong sign)
                'attention_entropy_trace': adversarial_trace.attention_entropy_trace,  # DEPRECATED (Phase 2 - weak)
                'semantic_drift_trace': adversarial_trace.semantic_drift_trace,  # PHASE 2a - PRIMARY SIGNAL
                'attack_id': i,
                'is_adversarial': True
            })

            # Validation for adversarial: high score = hallucination likely
            is_hallucination = attack['adversarial_score'] > 0.01
            validations.append({
                'is_hallucination': is_hallucination,
                'correctness_score': 1.0 - attack['adversarial_score'],  # Inverse relationship
                'adversarial_score': attack['adversarial_score'],
                'equivalence_score': attack['equivalence_score'],
                'attack_id': i
            })

        except Exception as e:
            logger.error(f"Error processing attack {i}: {e}")
            continue

    # Save traces
    logger.info(f"Saving {len(traces)} traces to {output_traces_path}")
    with open(output_traces_path, 'w') as f:
        json.dump(traces, f, indent=2)

    # Save validations
    logger.info(f"Saving {len(validations)} validations to {output_validation_path}")
    with open(output_validation_path, 'w') as f:
        json.dump(validations, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("TRACE GENERATION COMPLETE")
    print("="*80)
    print(f"\nInput: {attacks_path}")
    print(f"Output Traces: {output_traces_path}")
    print(f"Output Validation: {output_validation_path}")
    print(f"\nResults:")
    print(f"  - Input attacks: {len(attacks)}")
    print(f"  - Generated traces: {len(traces)} (2 per attack: original + adversarial)")
    print(f"  - Validation labels: {len(validations)}")
    print(f"\nSignals Computed (per trace):")
    print(f"  ✓ Entropy trace (token probability uncertainty) - Phase 2: VALIDATED")
    print(f"  ✓ Semantic Drift trace (cosine similarity to prompt) - Phase 2a: PRIMARY SIGNAL")
    print(f"  ✓ Attention trace (context engagement) - Phase 2: DEPRECATED (failed)")
    print(f"  ✓ Perplexity trace (exponential entropy) - Phase 2: DEPRECATED (wrong sign)")
    print(f"  ✓ Attention Entropy trace (scatteredness) - Phase 2: DEPRECATED (weak)")
    print(f"\nTrace Statistics:")
    hallucination_count = sum(1 for v in validations if v['is_hallucination'])
    normal_count = len(validations) - hallucination_count
    print(f"  - Hallucination traces: {hallucination_count}")
    print(f"  - Normal traces: {normal_count}")
    print(f"\nReady for Phase 2a Analysis:")
    print(f"  - Statistical analysis: python analysis/statistical_analysis.py")
    print(f"  - Multi-signal classifier: python analysis/multi_signal_classifier.py")
    print(f"  - Focus: Entropy + Semantic Drift (drop weak signals)")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate traces for SECA attacks'
    )
    parser.add_argument(
        '--attacks',
        type=str,
        required=True,
        help='Path to attacks JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save traces JSON'
    )
    parser.add_argument(
        '--validation',
        type=str,
        required=True,
        help='Path to save validation labels JSON'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Model to use for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Max new tokens to generate (default: 100)'
    )

    args = parser.parse_args()

    generate_traces_batch(
        attacks_path=args.attacks,
        output_traces_path=args.output,
        output_validation_path=args.validation,
        model_name=args.model,
        max_new_tokens=args.max_tokens
    )


if __name__ == '__main__':
    main()
