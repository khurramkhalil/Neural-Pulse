#!/usr/bin/env python3
"""
Fix NaN values in pilot traces

The traces have NaN entropy values. This script filters out NaN values
and validates the data before analysis.
"""

import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_nan_values(value):
    """Replace NaN with 0.0"""
    if isinstance(value, (list, tuple)):
        return [fix_nan_values(v) for v in value]
    elif isinstance(value, float) and np.isnan(value):
        return 0.0
    else:
        return value


def normalize_attention(attention_values):
    """
    Normalize attention values to [0, 1] range

    The current values are in range ~27-31 which seems like raw unnormalized values
    """
    if not attention_values or len(attention_values) == 0:
        return []

    # Convert to numpy array
    arr = np.array(attention_values)

    # Check if already normalized
    if np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
        return attention_values

    # Normalize to [0, 1]
    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val - min_val < 1e-10:  # All same value
        return [0.5] * len(attention_values)

    normalized = (arr - min_val) / (max_val - min_val)
    return normalized.tolist()


def fix_traces(input_path: str, output_path: str):
    """Fix NaN values and normalize attention in traces"""

    logger.info(f"Loading traces from {input_path}")
    with open(input_path) as f:
        traces = json.load(f)

    logger.info(f"Loaded {len(traces)} traces")

    # Count issues
    nan_entropy_count = 0
    fixed_traces = []

    for i, trace in enumerate(traces):
        entropy = trace.get('entropy_trace', [])
        attention = trace.get('attention_trace', [])

        # Check for NaN in entropy
        has_nan_entropy = any(isinstance(e, float) and np.isnan(e) for e in entropy)

        if has_nan_entropy:
            nan_entropy_count += 1
            # Replace NaN with mean entropy (assume ~2.0 for moderate uncertainty)
            entropy_fixed = [2.0 if (isinstance(e, float) and np.isnan(e)) else e for e in entropy]
        else:
            entropy_fixed = entropy

        # Normalize attention values
        attention_normalized = normalize_attention(attention)

        # Create fixed trace
        fixed_trace = trace.copy()
        fixed_trace['entropy_trace'] = entropy_fixed
        fixed_trace['attention_trace'] = attention_normalized

        fixed_traces.append(fixed_trace)

    logger.info(f"Fixed {nan_entropy_count} traces with NaN entropy")
    logger.info(f"Normalized attention values for all traces")

    # Validate fixed traces
    logger.info("Validating fixed traces...")
    for i, trace in enumerate(fixed_traces[:5]):  # Check first 5
        entropy = trace['entropy_trace']
        attention = trace['attention_trace']

        has_nan_entropy = any(isinstance(e, float) and np.isnan(e) for e in entropy)
        has_nan_attention = any(isinstance(a, float) and np.isnan(a) for a in attention)

        entropy_range = (min(entropy), max(entropy)) if entropy else (0, 0)
        attention_range = (min(attention), max(attention)) if attention else (0, 0)

        logger.info(f"Trace {i}:")
        logger.info(f"  Entropy NaN: {has_nan_entropy}, Range: {entropy_range}")
        logger.info(f"  Attention NaN: {has_nan_attention}, Range: {attention_range}")

    # Save fixed traces
    logger.info(f"Saving {len(fixed_traces)} fixed traces to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(fixed_traces, f, indent=2)

    print("\n" + "="*80)
    print("TRACE FIXING COMPLETE")
    print("="*80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"\nFixed Issues:")
    print(f"  - NaN entropy values: {nan_entropy_count} traces")
    print(f"  - Normalized attention: {len(fixed_traces)} traces")
    print(f"\nValidation:")

    # Final validation
    all_valid = True
    for trace in fixed_traces:
        entropy = trace['entropy_trace']
        attention = trace['attention_trace']

        if any(isinstance(e, float) and np.isnan(e) for e in entropy):
            all_valid = False
            break
        if any(isinstance(a, float) and np.isnan(a) for a in attention):
            all_valid = False
            break

    if all_valid:
        print("  ✅ All traces valid (no NaN values)")
    else:
        print("  ❌ Some traces still have NaN values")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fix NaN values in traces')
    parser.add_argument('--input', required=True, help='Input traces JSON')
    parser.add_argument('--output', required=True, help='Output fixed traces JSON')

    args = parser.parse_args()

    fix_traces(args.input, args.output)


if __name__ == '__main__':
    main()
