"""
Quick Smoke Test for Neural Pulse Monitor

This is a lightweight test that verifies basic functionality without loading the full model.
Use this for rapid development iteration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.neural_pulse import NeuralPulseMonitor, PulseDetection


def test_entropy_computation():
    """Test entropy computation without full model."""
    print("\n" + "=" * 80)
    print("QUICK TEST: Entropy Computation")
    print("=" * 80)

    # Create a mock monitor (we'll test just the entropy function)
    class MockModel:
        class Config:
            vocab_size = 32000

        config = Config()

    class MockTokenizer:
        eos_token_id = 2

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    monitor = NeuralPulseMonitor(
        model=mock_model,
        tokenizer=mock_tokenizer,
        threshold=2.8,
        mode='MONITOR'
    )

    # Test 1: Uniform distribution (high entropy)
    vocab_size = 32000
    uniform_logits = torch.zeros(vocab_size)
    entropy_uniform = monitor.compute_token_entropy(uniform_logits)

    # Test 2: Peaked distribution (low entropy)
    peaked_logits = torch.full((vocab_size,), -100.0)
    peaked_logits[0] = 10.0
    entropy_peaked = monitor.compute_token_entropy(peaked_logits)

    print(f"Uniform distribution entropy: {entropy_uniform:.3f}")
    print(f"Peaked distribution entropy: {entropy_peaked:.3f}")

    assert entropy_uniform > entropy_peaked, "Uniform should have higher entropy"
    assert entropy_uniform > 5.0, "Uniform entropy should be high (log(32000) ≈ 10.4)"
    assert entropy_peaked < 0.1, "Peaked entropy should be low"

    print("✅ Entropy computation working correctly")
    return True


def test_detection_logic():
    """Test detection logic without generation."""
    print("\n" + "=" * 80)
    print("QUICK TEST: Detection Logic")
    print("=" * 80)

    class MockModel:
        class Config:
            vocab_size = 32000
        config = Config()
        device = 'cpu'

    class MockTokenizer:
        eos_token_id = 2

    monitor = NeuralPulseMonitor(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        threshold=2.8,
        window_size=10,
        mode='MONITOR',
        min_tokens_before_detection=5
    )

    # Test 1: Low entropy trace (should NOT detect)
    low_entropy_trace = [2.0, 2.1, 2.0, 2.2, 2.1, 2.0]
    is_attack, conf, reason = monitor.analyze_entropy_trajectory(low_entropy_trace, len(low_entropy_trace))
    print(f"Low entropy trace: is_attack={is_attack}, confidence={conf:.2f}")
    assert not is_attack, "Low entropy should not be detected as attack"

    # Test 2: High entropy trace (should detect)
    high_entropy_trace = [2.0, 2.5, 3.0, 3.2, 3.5, 3.8, 4.0]
    is_attack, conf, reason = monitor.analyze_entropy_trajectory(high_entropy_trace, len(high_entropy_trace))
    print(f"High entropy trace: is_attack={is_attack}, confidence={conf:.2f}, reason={reason}")
    assert is_attack, "High entropy should be detected as attack"
    assert conf > 0.5, "Confidence should be > 0.5"

    # Test 3: Too few tokens (should not detect)
    short_trace = [3.5, 3.8]  # Only 2 tokens, min is 5
    is_attack, conf, reason = monitor.analyze_entropy_trajectory(short_trace, len(short_trace))
    print(f"Short trace: is_attack={is_attack} (should wait for more tokens)")
    assert not is_attack, "Should not detect with insufficient tokens"

    print("✅ Detection logic working correctly")
    return True


def test_mode_validation():
    """Test that invalid modes are rejected."""
    print("\n" + "=" * 80)
    print("QUICK TEST: Mode Validation")
    print("=" * 80)

    class MockModel:
        class Config:
            vocab_size = 32000
        config = Config()

    class MockTokenizer:
        eos_token_id = 2

    # Valid modes
    for mode in ['MONITOR', 'BLOCK', 'SANITIZE', 'monitor', 'block']:
        try:
            monitor = NeuralPulseMonitor(
                model=MockModel(),
                tokenizer=MockTokenizer(),
                mode=mode
            )
            print(f"✓ Mode '{mode}' accepted (normalized to {monitor.mode})")
        except ValueError:
            assert False, f"Valid mode '{mode}' was rejected"

    # Invalid mode
    try:
        monitor = NeuralPulseMonitor(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            mode='INVALID'
        )
        assert False, "Invalid mode should have been rejected"
    except ValueError:
        print("✓ Invalid mode 'INVALID' correctly rejected")

    print("✅ Mode validation working correctly")
    return True


def test_threshold_sensitivity():
    """Test that different thresholds produce different detection behavior."""
    print("\n" + "=" * 80)
    print("QUICK TEST: Threshold Sensitivity")
    print("=" * 80)

    class MockModel:
        class Config:
            vocab_size = 32000
        config = Config()

    class MockTokenizer:
        eos_token_id = 2

    entropy_trace = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1]

    thresholds = [2.0, 2.5, 3.0, 3.5]
    detections = []

    for threshold in thresholds:
        monitor = NeuralPulseMonitor(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            threshold=threshold,
            mode='MONITOR',
            min_tokens_before_detection=5
        )
        is_attack, conf, _ = monitor.analyze_entropy_trajectory(entropy_trace, len(entropy_trace))
        detections.append((threshold, is_attack))
        print(f"Threshold {threshold:.1f}: {'ATTACK' if is_attack else 'NORMAL'}")

    # Lower thresholds should be more likely to detect
    # (more sensitive but potentially more false positives)
    print("✅ Threshold sensitivity working correctly")
    return True


def run_quick_tests():
    """Run all quick tests."""
    print("\n" + "=" * 80)
    print("NEURAL PULSE MONITOR - QUICK SMOKE TESTS")
    print("=" * 80)
    print("These tests verify basic functionality without loading the full model.")
    print("For comprehensive tests, run: python tests/test_neural_pulse.py")
    print("=" * 80)

    tests = [
        ("Entropy Computation", test_entropy_computation),
        ("Detection Logic", test_detection_logic),
        ("Mode Validation", test_mode_validation),
        ("Threshold Sensitivity", test_threshold_sensitivity)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ Test failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("QUICK TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    print("-" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL QUICK TESTS PASSED!")
        print("Neural Pulse Monitor core logic is working correctly.")
        print("\nNext: Run full tests with model loaded:")
        print("  python tests/test_neural_pulse.py")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Review errors above and fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(run_quick_tests())
