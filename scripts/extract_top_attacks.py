#!/usr/bin/env python3
"""
Extract Top Attacks from Phase 1 Results

Extracts the most effective attacks (highest adversarial scores) for Phase 2 validation.

Usage:
    python scripts/extract_top_attacks.py \\
        --input seca_attacks_pilot_100.json \\
        --output datasets/top_attacks.json \\
        --threshold 0.01 \\
        --min-equivalence 0.85
"""

import json
import argparse
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_top_attacks(
    input_path: str,
    output_path: str,
    threshold: float = 0.01,
    min_equivalence: float = 0.85,
    top_n: int = None
) -> List[Dict]:
    """
    Extract top attacks based on adversarial score.

    Args:
        input_path: Path to Phase 1 results JSON
        output_path: Path to save filtered attacks
        threshold: Minimum adversarial score (higher = better attack)
        min_equivalence: Minimum semantic equivalence score
        top_n: If specified, only return top N attacks (sorted by score)

    Returns:
        List of top attack dictionaries
    """
    # Load results
    logger.info(f"Loading results from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    attacks = data['attacks']
    logger.info(f"Loaded {len(attacks)} total attacks")

    # Filter successful attacks
    successful_attacks = [a for a in attacks if a['success']]
    logger.info(f"Found {len(successful_attacks)} successful attacks")

    # Filter by thresholds
    filtered_attacks = [
        a for a in successful_attacks
        if a['adversarial_score'] > threshold and
           a['equivalence_score'] >= min_equivalence
    ]

    logger.info(f"After filtering (score > {threshold}, equiv >= {min_equivalence}): "
               f"{len(filtered_attacks)} attacks")

    # Sort by adversarial score DESCENDING (higher = better)
    filtered_attacks.sort(key=lambda x: x['adversarial_score'], reverse=True)

    # If top_n specified, only keep top N
    if top_n is not None:
        filtered_attacks = filtered_attacks[:top_n]
        logger.info(f"Keeping top {top_n} attacks")

    # Create output structure
    output_data = {
        'generator': data['generator'],
        'filter_criteria': {
            'min_adversarial_score': threshold,
            'min_equivalence_score': min_equivalence,
            'top_n': top_n
        },
        'statistics': {
            'total_input_attacks': len(attacks),
            'successful_attacks': len(successful_attacks),
            'filtered_attacks': len(filtered_attacks),
            'score_range': {
                'min': min(a['adversarial_score'] for a in filtered_attacks) if filtered_attacks else 0,
                'max': max(a['adversarial_score'] for a in filtered_attacks) if filtered_attacks else 0,
                'mean': sum(a['adversarial_score'] for a in filtered_attacks) / len(filtered_attacks) if filtered_attacks else 0
            },
            'equivalence_range': {
                'min': min(a['equivalence_score'] for a in filtered_attacks) if filtered_attacks else 0,
                'max': max(a['equivalence_score'] for a in filtered_attacks) if filtered_attacks else 0,
                'mean': sum(a['equivalence_score'] for a in filtered_attacks) / len(filtered_attacks) if filtered_attacks else 0
            }
        },
        'attacks': filtered_attacks
    }

    # Save to file
    logger.info(f"Saving {len(filtered_attacks)} attacks to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("TOP ATTACKS EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"\nFilter Criteria:")
    print(f"  - Adversarial Score > {threshold}")
    print(f"  - Equivalence Score >= {min_equivalence}")
    if top_n:
        print(f"  - Top {top_n} attacks only")

    print(f"\nResults:")
    print(f"  - Total input attacks: {len(attacks)}")
    print(f"  - Successful attacks: {len(successful_attacks)}")
    print(f"  - Filtered attacks: {len(filtered_attacks)}")

    if filtered_attacks:
        print(f"\nScore Statistics:")
        print(f"  - Min: {output_data['statistics']['score_range']['min']:.6f}")
        print(f"  - Max: {output_data['statistics']['score_range']['max']:.6f}")
        print(f"  - Mean: {output_data['statistics']['score_range']['mean']:.6f}")

        print(f"\nEquivalence Statistics:")
        print(f"  - Min: {output_data['statistics']['equivalence_range']['min']:.4f}")
        print(f"  - Max: {output_data['statistics']['equivalence_range']['max']:.4f}")
        print(f"  - Mean: {output_data['statistics']['equivalence_range']['mean']:.4f}")

        print(f"\nTop 5 Attacks:")
        for i, attack in enumerate(filtered_attacks[:5], 1):
            print(f"  {i}. Score: {attack['adversarial_score']:.6f}, "
                  f"Equiv: {attack['equivalence_score']:.4f}")
            print(f"     Prompt: {attack['original_prompt'][:80]}...")

    print("\n")

    return filtered_attacks


def main():
    parser = argparse.ArgumentParser(
        description='Extract top attacks from Phase 1 results'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to Phase 1 results JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save filtered attacks JSON'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='Minimum adversarial score (default: 0.01)'
    )
    parser.add_argument(
        '--min-equivalence',
        type=float,
        default=0.85,
        help='Minimum equivalence score (default: 0.85)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Only keep top N attacks (default: all above threshold)'
    )

    args = parser.parse_args()

    extract_top_attacks(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        min_equivalence=args.min_equivalence,
        top_n=args.top_n
    )


if __name__ == '__main__':
    main()
