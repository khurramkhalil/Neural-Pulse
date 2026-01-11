#!/usr/bin/env python3
"""
Test script to validate Phase 2 pipeline works correctly.

Tests all 3 components:
1. Statistical Analysis
2. Visualization
3. Formula Mining

Uses actual data from results/ directory.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_loading():
    """Test that we can load the data files"""
    print("="*80)
    print("TEST 1: Data Loading")
    print("="*80)

    traces_path = 'results/pilot_traces.json'
    validations_path = 'results/pilot_validation.json'

    print(f"\nLoading traces from: {traces_path}")
    with open(traces_path) as f:
        traces_data = json.load(f)

    print(f"  Type: {type(traces_data)}")

    # Handle both formats
    if isinstance(traces_data, dict):
        traces = traces_data.get('traces', traces_data)
        print(f"  Format: dict with 'traces' key")
    else:
        traces = traces_data
        print(f"  Format: list")

    print(f"  Traces count: {len(traces)}")

    # Check first trace structure
    if traces:
        trace = traces[0]
        print(f"\n  First trace keys: {list(trace.keys())}")
        print(f"  Has entropy_trace: {'entropy_trace' in trace}")
        print(f"  Has attention_trace: {'attention_trace' in trace}")
        if 'entropy_trace' in trace:
            print(f"  Entropy length: {len(trace['entropy_trace'])}")
            print(f"  First 5 entropy: {trace['entropy_trace'][:5]}")
        if 'attention_trace' in trace:
            print(f"  Attention length: {len(trace['attention_trace'])}")
            print(f"  First 5 attention: {trace['attention_trace'][:5]}")

    print(f"\nLoading validations from: {validations_path}")
    with open(validations_path) as f:
        validations_data = json.load(f)

    print(f"  Type: {type(validations_data)}")

    # Handle both formats
    if isinstance(validations_data, dict):
        validations = validations_data.get('validation_results', validations_data)
        print(f"  Format: dict with 'validation_results' key")
    else:
        validations = validations_data
        print(f"  Format: list")

    print(f"  Validations count: {len(validations)}")

    if validations:
        val = validations[0]
        print(f"\n  First validation keys: {list(val.keys())}")
        print(f"  Has is_hallucination: {'is_hallucination' in val}")

    # Check counts match
    assert len(traces) == len(validations), f"Count mismatch: {len(traces)} != {len(validations)}"
    print(f"\n✓ Data loading test PASSED")

    return traces, validations


def test_statistical_analysis(traces, validations):
    """Test statistical analysis component"""
    print("\n" + "="*80)
    print("TEST 2: Statistical Analysis")
    print("="*80)

    try:
        from analysis.statistical_analysis import SignalAnalyzer

        print("\n  Creating analyzer...")
        analyzer = SignalAnalyzer()

        print("  Running analysis...")
        results = analyzer.analyze_dataset(
            traces=traces,
            validations=validations,
            output_path=None  # Don't save
        )

        print("\n  Results keys:", list(results.keys()))

        if 'roc_curves' in results:
            print("\n  ROC Curves:")
            for signal, data in results['roc_curves'].items():
                print(f"    {signal}: AUC = {data['auc']:.3f}")

        if 'optimal_thresholds' in results:
            print("\n  Optimal Thresholds:")
            for signal, data in results['optimal_thresholds'].items():
                print(f"    {signal}: threshold = {data['threshold']:.3f}, F1 = {data['f1_score']:.3f}")

        print("\n✓ Statistical analysis test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Statistical analysis test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization(traces_path, validations_path):
    """Test visualization component"""
    print("\n" + "="*80)
    print("TEST 3: Visualization")
    print("="*80)

    try:
        from analysis.visualize_signals import SignalVisualizer
        import tempfile
        import shutil

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        print(f"\n  Using temp dir: {temp_dir}")

        print("  Creating visualizer...")
        visualizer = SignalVisualizer(output_dir=temp_dir)

        print("  Generating visualizations...")
        visualizer.generate_all_visualizations(
            traces_path=traces_path,
            validation_path=validations_path,
            prefix='test'
        )

        # Check that files were created
        files = os.listdir(temp_dir)
        print(f"\n  Generated {len(files)} files:")
        for f in sorted(files)[:5]:  # Show first 5
            print(f"    {f}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")

        # Cleanup
        shutil.rmtree(temp_dir)

        print("\n✓ Visualization test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Visualization test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_formula_mining(traces_path, validations_path):
    """Test formula mining component"""
    print("\n" + "="*80)
    print("TEST 4: Formula Mining")
    print("="*80)

    try:
        from analysis.formula_mining import FormulaMiner

        print("\n  Creating miner...")
        miner = FormulaMiner(
            theta_H_range=(1.5, 2.5, 0.5),  # Small range for testing
            theta_A_range=(0.3, 0.5, 0.1),
            T_range=(50, 50, 1),  # Single value
            w_range=(3, 3, 1),  # Single value
            metric='f1_score'
        )

        print("  Running formula mining (reduced search space)...")
        results = miner.mine_all_formulas(
            traces_path=traces_path,
            validation_path=validations_path,
            output_path=None,  # Don't save
            train_ratio=0.7
        )

        print("\n  Results:")
        for formula_type, data in results.items():
            print(f"    {formula_type}:")
            print(f"      Best params: {data['best_params']}")
            print(f"      Best F1: {data['best_f1_score']:.3f}")

        print("\n✓ Formula mining test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Formula mining test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("PHASE 2 PIPELINE VALIDATION")
    print("="*80)
    print("\nThis script tests that all Phase 2 components work correctly")
    print("with the actual data files.\n")

    traces_path = 'results/pilot_traces.json'
    validations_path = 'results/pilot_validation.json'

    # Check files exist
    if not os.path.exists(traces_path):
        print(f"✗ ERROR: {traces_path} not found")
        return 1
    if not os.path.exists(validations_path):
        print(f"✗ ERROR: {validations_path} not found")
        return 1

    # Run tests
    try:
        traces, validations = test_data_loading()
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        return 1

    results = []

    # Test 1: Statistical Analysis
    results.append(test_statistical_analysis(traces, validations))

    # Test 2: Visualization
    results.append(test_visualization(traces_path, validations_path))

    # Test 3: Formula Mining (skip if imports fail)
    try:
        results.append(test_formula_mining(traces_path, validations_path))
    except ImportError as e:
        print(f"\n⚠️  Skipping formula mining test (missing dependencies): {e}")
        results.append(None)  # Mark as skipped

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    test_names = [
        "Data Loading",
        "Statistical Analysis",
        "Visualization",
        "Formula Mining"
    ]

    passed = 0
    failed = 0
    skipped = 0

    for i, (name, result) in enumerate(zip(test_names, [True] + results)):
        if result is True:
            print(f"  ✓ {name}: PASSED")
            passed += 1
        elif result is False:
            print(f"  ✗ {name}: FAILED")
            failed += 1
        else:
            print(f"  ⚠️  {name}: SKIPPED")
            skipped += 1

    print()
    print(f"  Total: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 2 pipeline is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED! Please fix before running K8s job.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
