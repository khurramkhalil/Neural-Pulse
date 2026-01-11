"""
STL Formula Mining Module for Neural Pulse

Performs grid search over STL formula parameter space to find optimal thresholds:
- θ_H: Entropy threshold [1.5, 3.5] step 0.2
- θ_A: Attention threshold [0.2, 0.6] step 0.05
- T: Time horizon [50, 100] step 25
- w: Window size [3, 5] step 1

Optimizes for F1-score using cross-validation.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from core.stl_formulas import STLFormula, STLFormulaParams, evaluate_formula_on_dataset
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """Result from grid search"""
    formula_type: str
    best_params: Dict
    best_f1_score: float
    best_metrics: Dict
    all_results: List[Dict]  # All parameter combinations tried


class FormulaMiner:
    """
    Mines optimal STL formula parameters via grid search.

    Performs exhaustive search over parameter space and evaluates
    each combination on validation set.
    """

    def __init__(
        self,
        theta_H_range: Tuple[float, float, float] = (1.5, 3.5, 0.2),
        theta_A_range: Tuple[float, float, float] = (0.2, 0.6, 0.05),
        T_range: Tuple[int, int, int] = (50, 100, 25),
        w_range: Tuple[int, int, int] = (3, 5, 1),
        metric: str = 'f1_score'
    ):
        """
        Initialize formula miner.

        Args:
            theta_H_range: (start, stop, step) for entropy threshold
            theta_A_range: (start, stop, step) for attention threshold
            T_range: (start, stop, step) for time horizon
            w_range: (start, stop, step) for window size
            metric: Optimization metric ('f1_score', 'accuracy', 'tpr')
        """
        self.theta_H_values = np.arange(*theta_H_range).tolist()
        self.theta_A_values = np.arange(*theta_A_range).tolist()
        self.T_values = list(range(*T_range))
        self.w_values = list(range(*w_range))
        self.metric = metric

        logger.info(f"Formula miner initialized:")
        logger.info(f"  θ_H candidates: {len(self.theta_H_values)}")
        logger.info(f"  θ_A candidates: {len(self.theta_A_values)}")
        logger.info(f"  T candidates: {len(self.T_values)}")
        logger.info(f"  w candidates: {len(self.w_values)}")
        logger.info(f"  Optimization metric: {metric}")

    def grid_search_waffling(
        self,
        train_traces: List[Dict],
        train_validations: List[Dict],
        val_traces: Optional[List[Dict]] = None,
        val_validations: Optional[List[Dict]] = None
    ) -> GridSearchResult:
        """
        Grid search for φ₁ (waffling) formula parameters.

        Searches over (θ_H, T, w) space.

        Args:
            train_traces: Training traces
            train_validations: Training validation results
            val_traces: Optional validation traces (if None, uses train data)
            val_validations: Optional validation results

        Returns:
            GridSearchResult with best parameters
        """
        if val_traces is None:
            val_traces = train_traces
            val_validations = train_validations

        logger.info("Starting grid search for waffling formula...")

        best_score = -1.0
        best_params = None
        best_metrics = None
        all_results = []

        total_combinations = len(self.theta_H_values) * len(self.T_values) * len(self.w_values)

        with tqdm(total=total_combinations, desc="Grid search (waffling)") as pbar:
            for theta_H in self.theta_H_values:
                for T in self.T_values:
                    for w in self.w_values:
                        # Create formula
                        params = STLFormulaParams(theta_H=theta_H, T=T, w=w)
                        formula = STLFormula('waffling', params)

                        # Evaluate on validation set
                        results = evaluate_formula_on_dataset(formula, val_traces, val_validations)

                        score = results['metrics'][self.metric]

                        all_results.append({
                            'params': {'theta_H': theta_H, 'T': T, 'w': w},
                            'metrics': results['metrics']
                        })

                        if score > best_score:
                            best_score = score
                            best_params = {'theta_H': theta_H, 'T': T, 'w': w}
                            best_metrics = results['metrics']

                        pbar.update(1)

        logger.info(f"Best waffling params: {best_params}, {self.metric}={best_score:.3f}")

        return GridSearchResult(
            formula_type='waffling',
            best_params=best_params,
            best_f1_score=best_score,
            best_metrics=best_metrics,
            all_results=all_results
        )

    def grid_search_detachment(
        self,
        train_traces: List[Dict],
        train_validations: List[Dict],
        val_traces: Optional[List[Dict]] = None,
        val_validations: Optional[List[Dict]] = None
    ) -> GridSearchResult:
        """
        Grid search for φ₂ (detachment) formula parameters.

        Searches over (θ_A, T, w) space.

        Args:
            train_traces: Training traces
            train_validations: Training validation results
            val_traces: Optional validation traces
            val_validations: Optional validation results

        Returns:
            GridSearchResult with best parameters
        """
        if val_traces is None:
            val_traces = train_traces
            val_validations = train_validations

        logger.info("Starting grid search for detachment formula...")

        best_score = -1.0
        best_params = None
        best_metrics = None
        all_results = []

        total_combinations = len(self.theta_A_values) * len(self.T_values) * len(self.w_values)

        with tqdm(total=total_combinations, desc="Grid search (detachment)") as pbar:
            for theta_A in self.theta_A_values:
                for T in self.T_values:
                    for w in self.w_values:
                        params = STLFormulaParams(theta_A=theta_A, T=T, w=w)
                        formula = STLFormula('detachment', params)

                        results = evaluate_formula_on_dataset(formula, val_traces, val_validations)

                        score = results['metrics'][self.metric]

                        all_results.append({
                            'params': {'theta_A': theta_A, 'T': T, 'w': w},
                            'metrics': results['metrics']
                        })

                        if score > best_score:
                            best_score = score
                            best_params = {'theta_A': theta_A, 'T': T, 'w': w}
                            best_metrics = results['metrics']

                        pbar.update(1)

        logger.info(f"Best detachment params: {best_params}, {self.metric}={best_score:.3f}")

        return GridSearchResult(
            formula_type='detachment',
            best_params=best_params,
            best_f1_score=best_score,
            best_metrics=best_metrics,
            all_results=all_results
        )

    def grid_search_combined(
        self,
        train_traces: List[Dict],
        train_validations: List[Dict],
        val_traces: Optional[List[Dict]] = None,
        val_validations: Optional[List[Dict]] = None
    ) -> GridSearchResult:
        """
        Grid search for φ₃ (combined) formula parameters.

        Searches over (θ_H, θ_A, T, w) space.

        Args:
            train_traces: Training traces
            train_validations: Training validation results
            val_traces: Optional validation traces
            val_validations: Optional validation results

        Returns:
            GridSearchResult with best parameters
        """
        if val_traces is None:
            val_traces = train_traces
            val_validations = train_validations

        logger.info("Starting grid search for combined formula...")

        best_score = -1.0
        best_params = None
        best_metrics = None
        all_results = []

        total_combinations = (len(self.theta_H_values) * len(self.theta_A_values) *
                            len(self.T_values) * len(self.w_values))

        with tqdm(total=total_combinations, desc="Grid search (combined)") as pbar:
            for theta_H in self.theta_H_values:
                for theta_A in self.theta_A_values:
                    for T in self.T_values:
                        for w in self.w_values:
                            params = STLFormulaParams(theta_H=theta_H, theta_A=theta_A, T=T, w=w)
                            formula = STLFormula('combined', params)

                            results = evaluate_formula_on_dataset(formula, val_traces, val_validations)

                            score = results['metrics'][self.metric]

                            all_results.append({
                                'params': {'theta_H': theta_H, 'theta_A': theta_A, 'T': T, 'w': w},
                                'metrics': results['metrics']
                            })

                            if score > best_score:
                                best_score = score
                                best_params = {'theta_H': theta_H, 'theta_A': theta_A, 'T': T, 'w': w}
                                best_metrics = results['metrics']

                            pbar.update(1)

        logger.info(f"Best combined params: {best_params}, {self.metric}={best_score:.3f}")

        return GridSearchResult(
            formula_type='combined',
            best_params=best_params,
            best_f1_score=best_score,
            best_metrics=best_metrics,
            all_results=all_results
        )

    def mine_all_formulas(
        self,
        traces_path: str,
        validation_path: str,
        output_path: str,
        train_ratio: float = 0.7
    ):
        """
        Mine parameters for all three formulas.

        Args:
            traces_path: Path to traces JSON
            validation_path: Path to validation JSON
            output_path: Path to save results
            train_ratio: Ratio of data to use for training (rest for validation)
        """
        logger.info(f"Loading data from {traces_path}...")

        # Load data
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
            # Handle both formats: {'traces': [...]} or just [...]
            if isinstance(traces_data, dict):
                traces = traces_data.get('traces', traces_data)
            else:
                traces = traces_data

        with open(validation_path, 'r') as f:
            validation_data = json.load(f)
            # Handle both formats: {'validation_results': [...]} or just [...]
            if isinstance(validation_data, dict):
                validations = validation_data.get('validation_results', validation_data)
            else:
                validations = validation_data

        # Split train/val
        train_traces, val_traces, train_validations, val_validations = train_test_split(
            traces, validations, train_size=train_ratio, random_state=42
        )

        logger.info(f"Split data: {len(train_traces)} train, {len(val_traces)} val")

        # Mine all formulas
        results = {}

        # Waffling
        waffling_result = self.grid_search_waffling(
            train_traces, train_validations,
            val_traces, val_validations
        )
        results['waffling'] = {
            'best_params': waffling_result.best_params,
            'best_f1_score': waffling_result.best_f1_score,
            'best_metrics': waffling_result.best_metrics
        }

        # Detachment
        detachment_result = self.grid_search_detachment(
            train_traces, train_validations,
            val_traces, val_validations
        )
        results['detachment'] = {
            'best_params': detachment_result.best_params,
            'best_f1_score': detachment_result.best_f1_score,
            'best_metrics': detachment_result.best_metrics
        }

        # Combined
        combined_result = self.grid_search_combined(
            train_traces, train_validations,
            val_traces, val_validations
        )
        results['combined'] = {
            'best_params': combined_result.best_params,
            'best_f1_score': combined_result.best_f1_score,
            'best_metrics': combined_result.best_metrics
        }

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Formula mining complete! Results saved to {output_path}")

        return results


def main():
    """CLI interface for formula mining"""
    import argparse

    parser = argparse.ArgumentParser(description='STL Formula Mining')
    parser.add_argument('--traces', type=str, required=True, help='Path to traces JSON')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation JSON')
    parser.add_argument('--output', type=str, required=True, help='Path to output results JSON')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--metric', type=str, default='f1_score',
                       choices=['f1_score', 'accuracy', 'tpr'], help='Optimization metric')

    args = parser.parse_args()

    # Initialize miner
    miner = FormulaMiner(metric=args.metric)

    # Mine formulas
    results = miner.mine_all_formulas(
        traces_path=args.traces,
        validation_path=args.validation,
        output_path=args.output,
        train_ratio=args.train_ratio
    )

    # Print summary
    print("\n=== Formula Mining Results ===")
    for formula_type, result in results.items():
        print(f"\n{formula_type.upper()}:")
        print(f"  Best params: {result['best_params']}")
        print(f"  F1 score: {result['best_f1_score']:.3f}")
        print(f"  TPR: {result['best_metrics']['TPR']:.3f}")
        print(f"  FPR: {result['best_metrics']['FPR']:.3f}")
        print(f"  Precision: {result['best_metrics']['precision']:.3f}")


if __name__ == '__main__':
    main()
