#!/usr/bin/env python3
"""
Test script to verify the entropy fix works correctly.

This tests the fixed compute_entropy() method on synthetic data
to ensure it no longer produces NaN values.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llama_hook import LlamaSignalHook


def test_entropy_on_synthetic_logits():
    """Test entropy computation on various synthetic logits"""
    print("="*80)
    print("TESTING FIXED ENTROPY COMPUTATION")
    print("="*80 + "\n")

    # Create a hook instance (just to use its compute_entropy method)
    # We won't actually load the model
    hook = LlamaSignalHook.__new__(LlamaSignalHook)

    tests = [
        ("Uniform distribution (float32)", torch.ones(1000, dtype=torch.float32)),
        ("Uniform distribution (float16)", torch.ones(1000, dtype=torch.float16)),
        ("One-hot distribution (float32)", torch.cat([torch.tensor([100.0]), torch.zeros(999)])),
        ("One-hot distribution (float16)", torch.cat([torch.tensor([100.0], dtype=torch.float16), torch.zeros(999, dtype=torch.float16)])),
        ("Random logits (float32)", torch.randn(1000, dtype=torch.float32)),
        ("Random logits (float16)", torch.randn(1000, dtype=torch.float16)),
        ("Very peaked distribution (float32)", torch.cat([torch.tensor([1000.0]), torch.zeros(999)])),
        ("Very peaked distribution (float16)", torch.cat([torch.tensor([1000.0], dtype=torch.float16), torch.zeros(999, dtype=torch.float16)])),
    ]

    all_passed = True

    for name, logits in tests:
        print(f"Test: {name}")
        print(f"  Logits dtype: {logits.dtype}")
        print(f"  Logits shape: {logits.shape}")

        try:
            entropy = hook.compute_entropy(logits)

            is_nan = (entropy != entropy)  # NaN != NaN
            is_negative = entropy < 0
            is_finite = entropy != float('inf') and entropy != float('-inf')

            status = "✓ PASS" if (not is_nan and not is_negative and is_finite) else "✗ FAIL"

            print(f"  Entropy: {entropy:.4f}")
            print(f"  Is NaN: {is_nan}")
            print(f"  Is negative: {is_negative}")
            print(f"  Is finite: {is_finite}")
            print(f"  Status: {status}")

            if is_nan or is_negative or not is_finite:
                all_passed = False

        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            all_passed = False

        print()

    print("="*80)
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*80 + "\n")

    return all_passed


def test_edge_cases():
    """Test edge cases that might cause NaN"""
    print("="*80)
    print("TESTING EDGE CASES")
    print("="*80 + "\n")

    hook = LlamaSignalHook.__new__(LlamaSignalHook)

    edge_cases = [
        ("Very large logits (overflow test)", torch.tensor([1e10, 0.0, 0.0])),
        ("Very small logits (underflow test)", torch.tensor([1e-10, 1e-10, 1e-10])),
        ("Mixed scale logits", torch.tensor([1000.0, 1.0, 0.001, 1e-5])),
        ("All zeros", torch.zeros(100)),
        ("All same value", torch.ones(100) * 5.0),
        ("Single element", torch.tensor([1.0])),
        ("Two elements", torch.tensor([10.0, 1.0])),
    ]

    all_passed = True

    for name, logits in edge_cases:
        print(f"Edge case: {name}")
        print(f"  Logits: {logits[:5] if len(logits) > 5 else logits}")

        try:
            entropy = hook.compute_entropy(logits)

            is_valid = not (entropy != entropy) and entropy >= 0 and entropy != float('inf')
            status = "✓ PASS" if is_valid else "✗ FAIL"

            print(f"  Entropy: {entropy:.4f}")
            print(f"  Status: {status}")

            if not is_valid:
                all_passed = False

        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            all_passed = False

        print()

    print("="*80)
    print(f"EDGE CASES: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print("="*80 + "\n")

    return all_passed


def compare_with_old_implementation():
    """Compare new vs old implementation"""
    print("="*80)
    print("COMPARING OLD VS NEW IMPLEMENTATION")
    print("="*80 + "\n")

    def compute_entropy_old(logits):
        """Old buggy implementation"""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy

    hook = LlamaSignalHook.__new__(LlamaSignalHook)

    # Test on float16 (where bug manifests)
    logits_f16 = torch.randn(1000, dtype=torch.float16)

    print("Testing on random float16 logits (1000 elements):")
    print()

    try:
        old_entropy = compute_entropy_old(logits_f16)
        old_is_nan = (old_entropy != old_entropy)
        print(f"Old implementation:")
        print(f"  Result: {old_entropy:.4f}")
        print(f"  Is NaN: {old_is_nan}")
    except Exception as e:
        print(f"Old implementation: ✗ EXCEPTION: {e}")
        old_is_nan = True

    print()

    try:
        new_entropy = hook.compute_entropy(logits_f16)
        new_is_nan = (new_entropy != new_entropy)
        print(f"New implementation:")
        print(f"  Result: {new_entropy:.4f}")
        print(f"  Is NaN: {new_is_nan}")
    except Exception as e:
        print(f"New implementation: ✗ EXCEPTION: {e}")
        new_is_nan = True

    print()

    if old_is_nan and not new_is_nan:
        print("✓ FIX SUCCESSFUL: Old produced NaN, new produces valid value")
    elif not old_is_nan and not new_is_nan:
        print("✓ Both implementations work (bug may not manifest on this system)")
    else:
        print("✗ Issue not resolved")

    print()
    print("="*80 + "\n")


if __name__ == '__main__':
    print("\n")

    # Run all tests
    test1_passed = test_entropy_on_synthetic_logits()
    test2_passed = test_edge_cases()

    compare_with_old_implementation()

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Synthetic logits test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Edge cases test: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print()

    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED - Entropy fix is working correctly")
        print()
        print("NEXT STEPS:")
        print("  1. Regenerate traces: python scripts/generate_traces_batch.py \\")
        print("       --attacks seca_attacks_pilot_100.json \\")
        print("       --output datasets/pilot_traces_v2.json \\")
        print("       --validation datasets/pilot_validation_v2.json")
        print()
        print("  2. Run Phase 2 analysis on new traces")
        print("  3. Validate waffling signature hypothesis")
    else:
        print("✗ SOME TESTS FAILED - Fix needs more work")

    print("="*80)

    sys.exit(0 if (test1_passed and test2_passed) else 1)
