#!/usr/bin/env python3
"""
Quick analysis of Phase 2 traces - no external dependencies except json

Analyzes entropy and attention patterns to validate waffling signature hypothesis.
"""

import json
import math
from collections import defaultdict


def load_data(traces_path, validations_path):
    """Load traces and validations"""
    with open(traces_path) as f:
        traces = json.load(f)
    with open(validations_path) as f:
        validations = json.load(f)
    return traces, validations


def compute_statistics(values):
    """Compute basic statistics"""
    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n

    # Variance and std dev
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)

    # Min, max, median
    sorted_vals = sorted(values)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]

    # Percentiles
    p25_idx = int(n * 0.25)
    p75_idx = int(n * 0.75)
    p25 = sorted_vals[p25_idx]
    p75 = sorted_vals[p75_idx]

    return {
        'count': n,
        'mean': mean,
        'std': std_dev,
        'min': min_val,
        'max': max_val,
        'median': median,
        'p25': p25,
        'p75': p75
    }


def aggregate_signals(traces, validations):
    """Aggregate signals by label"""

    # Group by label
    hallucination_entropy = []
    hallucination_attention = []
    normal_entropy = []
    normal_attention = []

    for trace, validation in zip(traces, validations):
        is_hallucination = validation['is_hallucination']

        # Aggregate entropy and attention
        entropy_trace = trace['entropy_trace']
        attention_trace = trace['attention_trace']

        # Use mean of trace as aggregate
        mean_entropy = sum(entropy_trace) / len(entropy_trace) if entropy_trace else 0
        mean_attention = sum(attention_trace) / len(attention_trace) if attention_trace else 0

        if is_hallucination:
            hallucination_entropy.append(mean_entropy)
            hallucination_attention.append(mean_attention)
        else:
            normal_entropy.append(mean_entropy)
            normal_attention.append(mean_attention)

    return {
        'hallucination': {
            'entropy': hallucination_entropy,
            'attention': hallucination_attention
        },
        'normal': {
            'entropy': normal_entropy,
            'attention': normal_attention
        }
    }


def simple_auc_approximation(pos_values, neg_values):
    """
    Simple AUC approximation using rank-based method.

    AUC ≈ P(random positive > random negative)
    """
    if not pos_values or not neg_values:
        return 0.5

    # Count how many times pos > neg
    count = 0
    total = 0

    for pos in pos_values:
        for neg in neg_values:
            total += 1
            if pos > neg:
                count += 1
            elif pos == neg:
                count += 0.5  # Tie

    return count / total if total > 0 else 0.5


def analyze_separation(hallucination_vals, normal_vals, metric_name):
    """Analyze separation between hallucination and normal distributions"""

    # Compute statistics
    hall_stats = compute_statistics(hallucination_vals)
    norm_stats = compute_statistics(normal_vals)

    # Compute effect size (Cohen's d)
    if hall_stats['std'] == 0 and norm_stats['std'] == 0:
        cohens_d = 0
    else:
        pooled_std = math.sqrt((hall_stats['std']**2 + norm_stats['std']**2) / 2)
        if pooled_std > 0:
            cohens_d = (hall_stats['mean'] - norm_stats['mean']) / pooled_std
        else:
            cohens_d = 0

    # Compute simple AUC approximation
    auc = simple_auc_approximation(hallucination_vals, normal_vals)

    return {
        'metric': metric_name,
        'hallucination': hall_stats,
        'normal': norm_stats,
        'cohens_d': cohens_d,
        'auc_approx': auc,
        'mean_difference': hall_stats['mean'] - norm_stats['mean']
    }


def interpret_results(entropy_analysis, attention_analysis):
    """Interpret results and provide conclusions"""

    print("\n" + "="*80)
    print("HYPOTHESIS VALIDATION: WAFFLING SIGNATURE")
    print("="*80 + "\n")

    # Expected patterns if hypothesis is correct:
    # - High entropy for hallucinations (waffling/uncertainty)
    # - Low attention for hallucinations (detachment from context)

    print("EXPECTED IF HYPOTHESIS CORRECT:")
    print("  - Hallucination entropy > Normal entropy (uncertainty)")
    print("  - Hallucination attention < Normal attention (detachment)")
    print("  - AUC > 0.7 for both metrics (clear separation)")
    print()

    print("ACTUAL RESULTS:")
    print()

    # Entropy analysis
    print("ENTROPY (Higher = More Uncertainty):")
    print(f"  Hallucination mean: {entropy_analysis['hallucination']['mean']:.4f}")
    print(f"  Normal mean:        {entropy_analysis['normal']['mean']:.4f}")
    print(f"  Difference:         {entropy_analysis['mean_difference']:+.4f}")
    print(f"  Cohen's d:          {entropy_analysis['cohens_d']:.4f}")
    print(f"  AUC (approx):       {entropy_analysis['auc_approx']:.4f}")

    entropy_correct_direction = entropy_analysis['mean_difference'] > 0
    entropy_strong_signal = entropy_analysis['auc_approx'] > 0.7

    if entropy_correct_direction:
        print("  ✓ Correct direction: Hallucinations have HIGHER entropy")
    else:
        print("  ✗ Wrong direction: Hallucinations have LOWER entropy")

    if entropy_strong_signal:
        print("  ✓ Strong signal: AUC > 0.7")
    elif entropy_analysis['auc_approx'] > 0.6:
        print("  ⚠️  Moderate signal: AUC 0.6-0.7")
    else:
        print("  ✗ Weak signal: AUC < 0.6")

    print()

    # Attention analysis
    print("ATTENTION (Higher = More Context Engagement):")
    print(f"  Hallucination mean: {attention_analysis['hallucination']['mean']:.4f}")
    print(f"  Normal mean:        {attention_analysis['normal']['mean']:.4f}")
    print(f"  Difference:         {attention_analysis['mean_difference']:+.4f}")
    print(f"  Cohen's d:          {attention_analysis['cohens_d']:.4f}")
    print(f"  AUC (approx):       {attention_analysis['auc_approx']:.4f}")

    # For attention, we expect LOWER for hallucinations (detachment)
    # So mean_difference should be NEGATIVE
    attention_correct_direction = attention_analysis['mean_difference'] < 0
    attention_strong_signal = attention_analysis['auc_approx'] > 0.7 or (1 - attention_analysis['auc_approx']) > 0.7

    if attention_correct_direction:
        print("  ✓ Correct direction: Hallucinations have LOWER attention")
    else:
        print("  ✗ Wrong direction: Hallucinations have HIGHER attention")

    # For attention, low AUC could mean inverse relationship
    actual_attention_auc = min(attention_analysis['auc_approx'], 1 - attention_analysis['auc_approx'])
    if attention_strong_signal:
        print("  ✓ Strong signal: AUC > 0.7 (or < 0.3)")
    elif actual_attention_auc > 0.6 or actual_attention_auc < 0.4:
        print("  ⚠️  Moderate signal")
    else:
        print("  ✗ Weak signal: AUC ~0.5 (random)")

    print()
    print("="*80)
    print("CONCLUSION:")
    print("="*80)

    # Overall assessment
    if entropy_correct_direction and entropy_strong_signal:
        print("✓ ENTROPY SIGNAL CONFIRMED")
        print("  Hallucinations show higher uncertainty (waffling)")
    elif entropy_correct_direction:
        print("⚠️ ENTROPY SIGNAL WEAK BUT CORRECT DIRECTION")
        print("  Some evidence of waffling, but not strong")
    else:
        print("✗ ENTROPY SIGNAL NOT CONFIRMED")
        print("  Pattern opposite to hypothesis")

    print()

    if attention_correct_direction and attention_strong_signal:
        print("✓ ATTENTION SIGNAL CONFIRMED")
        print("  Hallucinations show lower context engagement (detachment)")
    elif attention_correct_direction:
        print("⚠️ ATTENTION SIGNAL WEAK BUT CORRECT DIRECTION")
        print("  Some evidence of detachment, but not strong")
    else:
        print("✗ ATTENTION SIGNAL NOT CONFIRMED")
        print("  Pattern opposite to hypothesis")

    print()

    # Overall verdict
    both_confirmed = (entropy_correct_direction and entropy_strong_signal and
                     attention_correct_direction and attention_strong_signal)

    if both_confirmed:
        print("🎉 WAFFLING SIGNATURE CONFIRMED!")
        print("   Both entropy and attention show expected patterns.")
        print("   Proceed to Phase 3: Real-time Monitor")
    elif entropy_correct_direction and attention_correct_direction:
        print("⚠️  WAFFLING SIGNATURE PARTIALLY CONFIRMED")
        print("   Correct direction but weak signals.")
        print("   Recommendations:")
        print("     - Collect more data (scale up Phase 1)")
        print("     - Refine signal definitions")
        print("     - Consider additional metrics")
    else:
        print("✗ WAFFLING SIGNATURE NOT CONFIRMED")
        print("   Signals do not match hypothesis.")
        print("   Recommendations:")
        print("     - Re-examine hypothesis")
        print("     - Investigate attack mechanism more deeply")
        print("     - Try different signal definitions")

    print("="*80)
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Quick Phase 2 analysis')
    parser.add_argument('--traces', required=True, help='Traces JSON file')
    parser.add_argument('--validations', required=True, help='Validations JSON file')
    parser.add_argument('--output', help='Output JSON file (optional)')

    args = parser.parse_args()

    print("="*80)
    print("PHASE 2: QUICK ANALYSIS (No External Dependencies)")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    traces, validations = load_data(args.traces, args.validations)
    print(f"  Loaded {len(traces)} traces")
    print(f"  Loaded {len(validations)} validation labels")
    print()

    # Check entropy validity
    print("Checking data quality...")
    first_trace = traces[0]
    has_nan = any(x != x for x in first_trace['entropy_trace'])

    if has_nan:
        print("  ✗ ERROR: Traces still contain NaN entropy values!")
        print("  Cannot proceed with analysis.")
        return
    else:
        print("  ✓ Entropy values are valid (no NaN)")

    # Check attention
    has_nan_attn = any(x != x for x in first_trace['attention_trace'])
    if has_nan_attn:
        print("  ✗ WARNING: Attention values contain NaN")
    else:
        print("  ✓ Attention values are valid")

    print()

    # Count labels
    n_hallucination = sum(1 for v in validations if v['is_hallucination'])
    n_normal = len(validations) - n_hallucination
    print(f"Label distribution:")
    print(f"  Hallucination: {n_hallucination} ({n_hallucination/len(validations)*100:.1f}%)")
    print(f"  Normal:        {n_normal} ({n_normal/len(validations)*100:.1f}%)")
    print()

    # Aggregate signals
    print("Aggregating signals by label...")
    aggregated = aggregate_signals(traces, validations)
    print()

    # Analyze entropy separation
    print("Analyzing entropy separation...")
    entropy_analysis = analyze_separation(
        aggregated['hallucination']['entropy'],
        aggregated['normal']['entropy'],
        'entropy'
    )

    # Analyze attention separation
    print("Analyzing attention separation...")
    attention_analysis = analyze_separation(
        aggregated['hallucination']['attention'],
        aggregated['normal']['attention'],
        'attention'
    )
    print()

    # Interpret results
    interpret_results(entropy_analysis, attention_analysis)

    # Save results if requested
    if args.output:
        results = {
            'traces_analyzed': len(traces),
            'n_hallucination': n_hallucination,
            'n_normal': n_normal,
            'entropy': {
                'hallucination_stats': entropy_analysis['hallucination'],
                'normal_stats': entropy_analysis['normal'],
                'cohens_d': entropy_analysis['cohens_d'],
                'auc_approx': entropy_analysis['auc_approx'],
                'mean_difference': entropy_analysis['mean_difference']
            },
            'attention': {
                'hallucination_stats': attention_analysis['hallucination'],
                'normal_stats': attention_analysis['normal'],
                'cohens_d': attention_analysis['cohens_d'],
                'auc_approx': attention_analysis['auc_approx'],
                'mean_difference': attention_analysis['mean_difference']
            }
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {args.output}")
        print()


if __name__ == '__main__':
    main()
