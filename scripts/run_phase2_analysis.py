#!/usr/bin/env python3
"""
Run Phase 2 Analysis on Generated Traces

This script runs the complete Phase 2 analysis pipeline on the generated traces.
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.statistical_analysis import SignalAnalyzer
from analysis.visualize_signals import SignalVisualizer
from analysis.formula_mining import FormulaMiner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_statistical_analysis(traces_path: str, validations_path: str, output_path: str):
    """Run statistical analysis"""
    logger.info("="*80)
    logger.info("STEP 1: STATISTICAL ANALYSIS")
    logger.info("="*80)

    # Load data
    logger.info(f"Loading traces from {traces_path}")
    with open(traces_path) as f:
        traces = json.load(f)

    logger.info(f"Loading validations from {validations_path}")
    with open(validations_path) as f:
        validations = json.load(f)

    logger.info(f"Loaded {len(traces)} traces and {len(validations)} validations")

    # Run analysis
    analyzer = SignalAnalyzer()
    results = analyzer.analyze_dataset(traces, validations, output_path=output_path)

    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    print(f"\nDataset:")
    print(f"  - Total traces: {results['dataset_summary']['total_traces']}")
    print(f"  - Attack traces: {results['dataset_summary']['attack_traces']}")
    print(f"  - Normal traces: {results['dataset_summary']['normal_traces']}")

    print(f"\nROC Analysis:")
    print(f"  - Entropy AUC: {results['roc_curves']['entropy']['auc']:.3f}")
    print(f"  - Attention AUC: {results['roc_curves']['attention']['auc']:.3f}")

    print(f"\nOptimal Thresholds:")
    print(f"  - Entropy threshold: {results['optimal_thresholds']['entropy']['threshold']:.3f}")
    print(f"    F1: {results['optimal_thresholds']['entropy']['f1_score']:.3f}")
    print(f"    TPR: {results['optimal_thresholds']['entropy']['tpr']:.3f}")
    print(f"    FPR: {results['optimal_thresholds']['entropy']['fpr']:.3f}")

    print(f"  - Attention threshold: {results['optimal_thresholds']['attention']['threshold']:.3f}")
    print(f"    F1: {results['optimal_thresholds']['attention']['f1_score']:.3f}")
    print(f"    TPR: {results['optimal_thresholds']['attention']['tpr']:.3f}")
    print(f"    FPR: {results['optimal_thresholds']['attention']['fpr']:.3f}")

    if results['roc_curves']['entropy']['auc'] > 0.7:
        print("\n✅ WAFFLING SIGNATURE CONFIRMED!")
        print("   High-score attacks show distinct entropy patterns.")
    else:
        print("\n⚠️ WAFFLING SIGNATURE UNCLEAR")
        print("   May need more data or different signals.")

    print(f"\nResults saved to: {output_path}\n")

    return results


def run_visualizations(traces_path: str, validations_path: str, output_dir: str):
    """Generate visualizations"""
    logger.info("="*80)
    logger.info("STEP 2: GENERATING VISUALIZATIONS")
    logger.info("="*80)

    # Load data
    with open(traces_path) as f:
        traces = json.load(f)

    with open(validations_path) as f:
        validations = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    visualizer = SignalVisualizer()
    visualizer.generate_all_visualizations(
        traces_path=traces_path,
        validation_path=validations_path,
        prefix='phase2'
    )
    # visualizer.generate_all_visualizations(
    #     traces_path,
    #     validations_path,
    #     output_dir=output_dir,
    #     prefix='phase2'
    # )

    print("\n" + "="*80)
    print("VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print("  - phase2_entropy_distribution.png")
    print("  - phase2_attention_distribution.png")
    print("  - phase2_attack_traces_sample.png")
    print("  - phase2_normal_traces_sample.png")
    print("  - phase2_entropy_heatmap.png")
    print("  - phase2_attention_heatmap.png")
    print("  - phase2_statistical_summary.png")
    print()


def run_formula_mining(traces_path: str, validations_path: str, output_path: str):
    """Run STL formula mining"""
    logger.info("="*80)
    logger.info("STEP 3: STL FORMULA MINING")
    logger.info("="*80)

    # Load data
    with open(traces_path) as f:
        traces = json.load(f)

    with open(validations_path) as f:
        validations = json.load(f)

    # Run mining
    miner = FormulaMiner(
        theta_H_range=(1.5, 3.5, 0.2),
        theta_A_range=(0.2, 0.6, 0.05),
        T_range=(50, 100, 25),
        w_range=(3, 5, 1),
        metric='f1_score'
    )

    results = miner.mine_all_formulas(
        traces,
        validations,
        output_path=output_path,
        train_ratio=0.7
    )

    print("\n" + "="*80)
    print("FORMULA MINING RESULTS")
    print("="*80)

    for formula_type, result in results.items():
        print(f"\n{formula_type.upper()}:")
        print(f"  Best Params: {result['best_params']}")
        print(f"  F1 Score: {result['best_f1_score']:.3f}")
        print(f"  TPR: {result['best_metrics']['TPR']:.3f}")
        print(f"  FPR: {result['best_metrics']['FPR']:.3f}")
        print(f"  Precision: {result['best_metrics']['precision']:.3f}")
        print(f"  Recall: {result['best_metrics']['recall']:.3f}")

    print(f"\nResults saved to: {output_path}\n")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Phase 2 analysis pipeline')
    parser.add_argument('--traces', required=True, help='Path to traces JSON')
    parser.add_argument('--validations', required=True, help='Path to validations JSON')
    parser.add_argument('--output-dir', default='results/phase2', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Step 1: Statistical Analysis
        stats_results = run_statistical_analysis(
            traces_path=args.traces,
            validations_path=args.validations,
            output_path=os.path.join(args.output_dir, 'phase2_statistics.json')
        )

        # Step 2: Visualizations
        run_visualizations(
            traces_path=args.traces,
            validations_path=args.validations,
            output_dir=os.path.join(args.output_dir, 'figures')
        )

        # Step 3: Formula Mining
        formula_results = run_formula_mining(
            traces_path=args.traces,
            validations_path=args.validations,
            output_path=os.path.join(args.output_dir, 'phase2_formula_mining.json')
        )

        # Final Summary
        print("\n" + "="*80)
        print("PHASE 2 ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {args.output_dir}")
        print("\nKey Findings:")
        print(f"  - Entropy AUC: {stats_results['roc_curves']['entropy']['auc']:.3f}")
        print(f"  - Attention AUC: {stats_results['roc_curves']['attention']['auc']:.3f}")
        print(f"  - Best Formula: {formula_results['combined']['best_params']}")
        print(f"  - Best F1 Score: {formula_results['combined']['best_f1_score']:.3f}")

        if stats_results['roc_curves']['entropy']['auc'] > 0.7:
            print("\n✅ SUCCESS: Waffling signature validated!")
            print("   → Proceed to Phase 3: Real-time Monitor")
        else:
            print("\n⚠️ UNCLEAR: Waffling signature not strongly confirmed")
            print("   → Consider collecting more data or refining signals")

        print()

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
