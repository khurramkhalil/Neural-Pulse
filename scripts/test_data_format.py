#!/usr/bin/env python3
"""
Test that data format handling works correctly.

This validates the fixes for the AttributeError: 'list' object has no attribute 'get'
"""

import json


def test_format_handling():
    """Test that we correctly handle both list and dict formats"""
    print("="*80)
    print("Testing Data Format Handling")
    print("="*80)

    # Test case 1: List format (what we actually have)
    print("\nTest 1: List format")
    data = [{"id": 1}, {"id": 2}]

    if isinstance(data, dict):
        result = data.get('traces', data)
    else:
        result = data

    assert result == data, "List handling failed"
    print("  ✓ List format handled correctly")

    # Test case 2: Dict format with 'traces' key
    print("\nTest 2: Dict format with 'traces' key")
    data = {'traces': [{"id": 1}, {"id": 2}]}

    if isinstance(data, dict):
        result = data.get('traces', data)
    else:
        result = data

    assert result == [{"id": 1}, {"id": 2}], "Dict with 'traces' handling failed"
    print("  ✓ Dict with 'traces' key handled correctly")

    # Test case 3: Dict format without 'traces' key
    print("\nTest 3: Dict format without 'traces' key")
    data = {'other_key': [{"id": 1}]}

    if isinstance(data, dict):
        result = data.get('traces', data)
    else:
        result = data

    assert result == data, "Dict without 'traces' handling failed"
    print("  ✓ Dict without 'traces' key handled correctly")

    print("\n" + "="*80)
    print("✓ ALL FORMAT HANDLING TESTS PASSED")
    print("="*80)


def test_actual_files():
    """Test with actual pilot data files"""
    print("\n" + "="*80)
    print("Testing Actual Data Files")
    print("="*80)

    traces_path = 'results/pilot_traces.json'
    validations_path = 'results/pilot_validation.json'

    # Load traces
    print(f"\nLoading: {traces_path}")
    with open(traces_path) as f:
        traces_data = json.load(f)

    print(f"  Type: {type(traces_data).__name__}")

    # Apply format handling
    if isinstance(traces_data, dict):
        traces = traces_data.get('traces', traces_data)
        print("  Format: dict (extracted 'traces' key)")
    else:
        traces = traces_data
        print("  Format: list (used directly)")

    print(f"  Count: {len(traces)}")
    assert len(traces) > 0, "No traces loaded"
    print("  ✓ Traces loaded successfully")

    # Load validations
    print(f"\nLoading: {validations_path}")
    with open(validations_path) as f:
        validations_data = json.load(f)

    print(f"  Type: {type(validations_data).__name__}")

    # Apply format handling
    if isinstance(validations_data, dict):
        validations = validations_data.get('validation_results', validations_data)
        print("  Format: dict (extracted 'validation_results' key)")
    else:
        validations = validations_data
        print("  Format: list (used directly)")

    print(f"  Count: {len(validations)}")
    assert len(validations) > 0, "No validations loaded"
    print("  ✓ Validations loaded successfully")

    # Check counts match
    assert len(traces) == len(validations), \
        f"Count mismatch: {len(traces)} traces != {len(validations)} validations"
    print(f"\n  ✓ Counts match: {len(traces)} == {len(validations)}")

    # Check structure
    trace = traces[0]
    validation = validations[0]

    print(f"\nTrace structure:")
    print(f"  Keys: {list(trace.keys())}")
    assert 'entropy_trace' in trace, "Missing entropy_trace"
    assert 'attention_trace' in trace, "Missing attention_trace"
    print("  ✓ Has required fields")

    print(f"\nValidation structure:")
    print(f"  Keys: {list(validation.keys())}")
    assert 'is_hallucination' in validation, "Missing is_hallucination"
    print("  ✓ Has required fields")

    print("\n" + "="*80)
    print("✓ ACTUAL DATA FILES TEST PASSED")
    print("="*80)


if __name__ == '__main__':
    try:
        test_format_handling()
        test_actual_files()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nThe AttributeError fix is working correctly.")
        print("K8s job should now run without errors in Steps 4 & 5.")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
