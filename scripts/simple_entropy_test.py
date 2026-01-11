#!/usr/bin/env python3
"""
Simplified entropy test - identifies the NaN bug source
"""

import json
import math


def analyze_trace_file():
    """Analyze the actual trace file to understand the NaN pattern"""
    print("="*80)
    print("ANALYZING PILOT TRACES FOR NaN PATTERN")
    print("="*80 + "\n")

    with open('results/pilot_traces.json') as f:
        traces = json.load(f)

    print(f"Total traces: {len(traces)}")
    print(f"Analyzing first 5 traces...\n")

    for i in range(min(5, len(traces))):
        trace = traces[i]
        entropy = trace['entropy_trace']
        attention = trace['attention_trace']

        all_entropy_nan = all(x != x for x in entropy)  # NaN != NaN
        any_attention_nan = any(x != x for x in attention)

        print(f"Trace {i}:")
        print(f"  Prompt: {trace['prompt'][:60]}...")
        print(f"  Generated: {trace['generated_text'][:60]}...")
        print(f"  Entropy length: {len(entropy)}")
        print(f"  Attention length: {len(attention)}")
        print(f"  All entropy NaN: {all_entropy_nan}")
        print(f"  Any attention NaN: {any_attention_nan}")

        if not all_entropy_nan:
            print(f"  Entropy sample: {entropy[:5]}")
        if not any_attention_nan:
            print(f"  Attention range: [{min(attention):.2f}, {max(attention):.2f}]")

        print()

    # Check if ALL traces have NaN entropy
    all_traces_nan = all(
        all(x != x for x in trace['entropy_trace'])
        for trace in traces
    )

    print(f"ALL {len(traces)} traces have NaN entropy: {all_traces_nan}")
    print()

    # This suggests a systematic bug in the entropy computation
    print("DIAGNOSIS:")
    print("-" * 80)
    if all_traces_nan:
        print("✗ SYSTEMATIC BUG: Every single trace has NaN entropy")
        print("  This indicates the bug is in the compute_entropy() function itself,")
        print("  not in the input data.")
        print()
        print("LIKELY CAUSES:")
        print("  1. Float16 precision issue (if running on GPU)")
        print("  2. Logits tensor dimension mismatch")
        print("  3. Softmax computation on wrong dimension")
        print("  4. NaN propagation from model output")
    else:
        print("✓ Some traces have valid entropy - bug is intermittent")

    print("="*80)


def test_entropy_computation_logic():
    """Test the entropy formula on known values"""
    print("\n" + "="*80)
    print("TESTING ENTROPY COMPUTATION LOGIC")
    print("="*80 + "\n")

    # Simulate what compute_entropy does

    print("Test 1: Simple uniform distribution")
    print("-" * 40)
    # Simulating logits = [1, 1, 1, 1]
    logits = [1.0, 1.0, 1.0, 1.0]

    # Softmax: e^1 / (4 * e^1) = 0.25 for each
    exp_logits = [math.exp(x) for x in logits]
    sum_exp = sum(exp_logits)
    probs = [x / sum_exp for x in exp_logits]
    print(f"  Logits: {logits}")
    print(f"  Probs: {[f'{p:.4f}' for p in probs]}")
    print(f"  Sum of probs: {sum(probs):.6f}")

    # Entropy: -sum(p * log(p))
    epsilon = 1e-10
    probs_eps = [p + epsilon for p in probs]
    log_probs = [math.log(p) for p in probs_eps]
    products = [p * lp for p, lp in zip(probs, log_probs)]
    entropy = -sum(products)

    print(f"  Log probs: {[f'{lp:.4f}' for lp in log_probs]}")
    print(f"  Products: {[f'{prod:.4f}' for prod in products]}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Expected: {math.log(4):.4f} (log(4) for uniform over 4 items)")
    print(f"  Is NaN: {entropy != entropy}")
    print()

    print("Test 2: One-hot distribution")
    print("-" * 40)
    # Simulating logits = [100, 0, 0, 0]
    logits = [100.0, 0.0, 0.0, 0.0]

    exp_logits = [math.exp(x - 100) for x in logits]  # Shift for numerical stability
    sum_exp = sum(exp_logits)
    probs = [x / sum_exp for x in exp_logits]
    print(f"  Logits: {logits}")
    print(f"  Probs: {[f'{p:.10f}' for p in probs]}")

    probs_eps = [p + epsilon for p in probs]
    log_probs = [math.log(p) if p > 0 else math.log(epsilon) for p in probs_eps]
    products = [p * lp for p, lp in zip(probs, log_probs)]
    entropy = -sum(products)

    print(f"  Entropy: {entropy:.4f}")
    print(f"  Expected: ~0.0 (certain outcome)")
    print(f"  Is NaN: {entropy != entropy}")
    print()

    print("Test 3: Testing with very small probabilities (potential NaN source)")
    print("-" * 40)
    # Test the edge case: p * log(p + epsilon) when p is very small
    small_p = 1e-20
    log_small = math.log(small_p + epsilon)
    product = small_p * log_small

    print(f"  Very small prob: {small_p}")
    print(f"  log(p + epsilon): {log_small:.4f}")
    print(f"  p * log(p + epsilon): {product:.10e}")
    print(f"  Is NaN: {product != product}")
    print(f"  Is finite: {math.isfinite(product)}")
    print()

    # The issue might be with 0 * -inf
    zero_prob = 0.0
    log_zero_eps = math.log(zero_prob + epsilon)
    product_zero = zero_prob * log_zero_eps

    print(f"  Zero prob: {zero_prob}")
    print(f"  log(0 + epsilon): {log_zero_eps:.4f}")
    print(f"  0 * log(epsilon): {product_zero:.10e}")
    print(f"  Is NaN: {product_zero != product_zero}")
    print(f"  Note: 0 * log(epsilon) = 0 * (-large number) = 0 (should be fine)")

    print("\n" + "="*80)


if __name__ == '__main__':
    test_entropy_computation_logic()
    analyze_trace_file()

    print("\nNEXT STEPS:")
    print("-" * 80)
    print("1. Check if model was run with float16 (GPU) vs float32 (CPU)")
    print("2. Examine the actual LlamaHook code for dimension issues")
    print("3. Add debug logging directly in core/llama_hook.py")
    print("4. Test with a simple prompt locally to reproduce the bug")
    print()
