"""
STL Formula Implementation for Neural Pulse

Implements three Signal Temporal Logic formulas for detecting SECA attacks:

φ₁: Eventually[0,T](Always[t,t+w](H(t) > θ_H))
    Waffling - Sustained high entropy

φ₂: Eventually[0,T](Always[t,t+w](A(t) < θ_A))
    Detachment - Sustained low context attention

φ₃: Eventually[0,T](Always[t,t+w](H(t) > θ_H AND A(t) < θ_A))
    Combined - Both signals simultaneously

Uses rtamt library for STL evaluation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class STLFormulaParams:
    """Parameters for STL formulas"""
    theta_H: Optional[float] = None  # Entropy threshold
    theta_A: Optional[float] = None  # Attention threshold
    T: int = 100  # Maximum time horizon
    w: int = 3    # Window size for "Always" (sustained period)


@dataclass
class STLEvaluationResult:
    """Result of STL formula evaluation"""
    formula_name: str
    is_violation: bool  # True if formula detected attack
    robustness_score: float  # Quantitative robustness (how much violation)
    violation_time: Optional[int] = None  # Time step where violation occurred


class STLFormula:
    """
    STL Formula evaluator.

    Implements temporal logic formulas without rtamt dependency for simplicity.
    Uses direct boolean evaluation over signal traces.
    """

    def __init__(self, formula_type: str, params: STLFormulaParams):
        """
        Initialize STL formula.

        Args:
            formula_type: 'waffling', 'detachment', or 'combined'
            params: STLFormulaParams with thresholds and time parameters
        """
        self.formula_type = formula_type
        self.params = params

        if formula_type == 'waffling' and params.theta_H is None:
            raise ValueError("theta_H required for waffling formula")
        if formula_type == 'detachment' and params.theta_A is None:
            raise ValueError("theta_A required for detachment formula")
        if formula_type == 'combined':
            if params.theta_H is None or params.theta_A is None:
                raise ValueError("Both theta_H and theta_A required for combined formula")

        logger.info(f"Initialized {formula_type} formula with params: {params}")

    def evaluate_waffling(
        self,
        entropy_trace: List[float]
    ) -> STLEvaluationResult:
        """
        Evaluate φ₁: Eventually[0,T](Always[t,t+w](H(t) > θ_H))

        Detects sustained high entropy (waffling).

        Args:
            entropy_trace: Entropy signal H(t)

        Returns:
            STLEvaluationResult
        """
        if len(entropy_trace) == 0:
            return STLEvaluationResult(
                formula_name='waffling',
                is_violation=False,
                robustness_score=-np.inf
            )

        theta_H = self.params.theta_H
        w = self.params.w
        T = min(self.params.T, len(entropy_trace))

        # Check for sustained high entropy (w consecutive tokens above threshold)
        violation_detected = False
        max_robustness = -np.inf
        violation_time = None

        for t in range(T - w + 1):
            window = entropy_trace[t:t+w]

            # Check if ALL tokens in window exceed threshold
            if all(h > theta_H for h in window):
                violation_detected = True
                # Robustness: minimum excess above threshold in window
                robustness = min(h - theta_H for h in window)
                if robustness > max_robustness:
                    max_robustness = robustness
                    violation_time = t

        if not violation_detected:
            # Compute how close we came (max entropy seen)
            max_robustness = max(entropy_trace[:T]) - theta_H

        return STLEvaluationResult(
            formula_name='waffling',
            is_violation=violation_detected,
            robustness_score=max_robustness,
            violation_time=violation_time
        )

    def evaluate_detachment(
        self,
        attention_trace: List[float]
    ) -> STLEvaluationResult:
        """
        Evaluate φ₂: Eventually[0,T](Always[t,t+w](A(t) < θ_A))

        Detects sustained low context attention (detachment).

        Args:
            attention_trace: Attention signal A(t)

        Returns:
            STLEvaluationResult
        """
        if len(attention_trace) == 0:
            return STLEvaluationResult(
                formula_name='detachment',
                is_violation=False,
                robustness_score=-np.inf
            )

        theta_A = self.params.theta_A
        w = self.params.w
        T = min(self.params.T, len(attention_trace))

        violation_detected = False
        max_robustness = -np.inf
        violation_time = None

        for t in range(T - w + 1):
            window = attention_trace[t:t+w]

            # Check if ALL tokens in window are below threshold
            if all(a < theta_A for a in window):
                violation_detected = True
                # Robustness: minimum deficit below threshold in window
                robustness = min(theta_A - a for a in window)
                if robustness > max_robustness:
                    max_robustness = robustness
                    violation_time = t

        if not violation_detected:
            # Compute how close we came (min attention seen)
            max_robustness = theta_A - min(attention_trace[:T])

        return STLEvaluationResult(
            formula_name='detachment',
            is_violation=violation_detected,
            robustness_score=max_robustness,
            violation_time=violation_time
        )

    def evaluate_combined(
        self,
        entropy_trace: List[float],
        attention_trace: List[float]
    ) -> STLEvaluationResult:
        """
        Evaluate φ₃: Eventually[0,T](Always[t,t+w](H(t) > θ_H AND A(t) < θ_A))

        Detects simultaneous waffling AND detachment.

        Args:
            entropy_trace: Entropy signal H(t)
            attention_trace: Attention signal A(t)

        Returns:
            STLEvaluationResult
        """
        if len(entropy_trace) == 0 or len(attention_trace) == 0:
            return STLEvaluationResult(
                formula_name='combined',
                is_violation=False,
                robustness_score=-np.inf
            )

        theta_H = self.params.theta_H
        theta_A = self.params.theta_A
        w = self.params.w
        T = min(self.params.T, len(entropy_trace), len(attention_trace))

        violation_detected = False
        max_robustness = -np.inf
        violation_time = None

        for t in range(T - w + 1):
            entropy_window = entropy_trace[t:t+w]
            attention_window = attention_trace[t:t+w]

            # Check if BOTH conditions hold for ALL tokens in window
            entropy_condition = all(h > theta_H for h in entropy_window)
            attention_condition = all(a < theta_A for a in attention_window)

            if entropy_condition and attention_condition:
                violation_detected = True
                # Robustness: minimum of both conditions
                robustness_h = min(h - theta_H for h in entropy_window)
                robustness_a = min(theta_A - a for a in attention_window)
                robustness = min(robustness_h, robustness_a)

                if robustness > max_robustness:
                    max_robustness = robustness
                    violation_time = t

        if not violation_detected:
            # Compute how close we came (worse of the two conditions)
            max_robustness = -np.inf

        return STLEvaluationResult(
            formula_name='combined',
            is_violation=violation_detected,
            robustness_score=max_robustness,
            violation_time=violation_time
        )

    def evaluate(
        self,
        entropy_trace: Optional[List[float]] = None,
        attention_trace: Optional[List[float]] = None
    ) -> STLEvaluationResult:
        """
        Evaluate formula on given traces.

        Args:
            entropy_trace: Entropy signal (required for waffling, combined)
            attention_trace: Attention signal (required for detachment, combined)

        Returns:
            STLEvaluationResult
        """
        if self.formula_type == 'waffling':
            if entropy_trace is None:
                raise ValueError("Entropy trace required for waffling formula")
            return self.evaluate_waffling(entropy_trace)

        elif self.formula_type == 'detachment':
            if attention_trace is None:
                raise ValueError("Attention trace required for detachment formula")
            return self.evaluate_detachment(attention_trace)

        elif self.formula_type == 'combined':
            if entropy_trace is None or attention_trace is None:
                raise ValueError("Both traces required for combined formula")
            return self.evaluate_combined(entropy_trace, attention_trace)

        else:
            raise ValueError(f"Unknown formula type: {self.formula_type}")


def evaluate_formula_on_dataset(
    formula: STLFormula,
    traces: List[Dict],
    validations: List[Dict]
) -> Dict:
    """
    Evaluate STL formula on entire dataset.

    Args:
        formula: STLFormula instance
        traces: List of trace dicts with entropy/attention signals
        validations: List of validation dicts with is_hallucination labels

    Returns:
        Dict with evaluation metrics
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for trace, validation in zip(traces, validations):
        entropy_trace = trace.get('entropy_trace', [])
        attention_trace = trace.get('attention_trace', [])

        # Evaluate formula
        result = formula.evaluate(
            entropy_trace=entropy_trace if len(entropy_trace) > 0 else None,
            attention_trace=attention_trace if len(attention_trace) > 0 else None
        )

        # Compare with ground truth
        is_attack = validation['is_hallucination']
        predicted_attack = result.is_violation

        if is_attack and predicted_attack:
            true_positives += 1
        elif is_attack and not predicted_attack:
            false_negatives += 1
        elif not is_attack and predicted_attack:
            false_positives += 1
        else:
            true_negatives += 1

    # Compute metrics
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return {
        'formula_type': formula.formula_type,
        'parameters': {
            'theta_H': formula.params.theta_H,
            'theta_A': formula.params.theta_A,
            'T': formula.params.T,
            'w': formula.params.w
        },
        'confusion_matrix': {
            'TP': true_positives,
            'FP': false_positives,
            'TN': true_negatives,
            'FN': false_negatives
        },
        'metrics': {
            'TPR': tpr,
            'FPR': fpr,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }
    }


def main():
    """CLI interface for STL formula evaluation"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='STL Formula Evaluation')
    parser.add_argument('--traces', type=str, required=True, help='Path to traces JSON')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation JSON')
    parser.add_argument('--formula', type=str, required=True,
                       choices=['waffling', 'detachment', 'combined'], help='Formula type')
    parser.add_argument('--theta_H', type=float, help='Entropy threshold')
    parser.add_argument('--theta_A', type=float, help='Attention threshold')
    parser.add_argument('--T', type=int, default=100, help='Time horizon')
    parser.add_argument('--w', type=int, default=3, help='Window size')

    args = parser.parse_args()

    # Load data
    with open(args.traces, 'r') as f:
        traces_data = json.load(f)
        traces = traces_data.get('traces', traces_data)

    with open(args.validation, 'r') as f:
        validation_data = json.load(f)
        validations = validation_data.get('validation_results', validation_data)

    # Create formula
    params = STLFormulaParams(
        theta_H=args.theta_H,
        theta_A=args.theta_A,
        T=args.T,
        w=args.w
    )
    formula = STLFormula(args.formula, params)

    # Evaluate
    results = evaluate_formula_on_dataset(formula, traces, validations)

    # Print results
    print(f"\n=== STL Formula Evaluation: {args.formula} ===")
    print(f"Parameters: θ_H={args.theta_H}, θ_A={args.theta_A}, T={args.T}, w={args.w}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={results['confusion_matrix']['TP']}, FP={results['confusion_matrix']['FP']}")
    print(f"  TN={results['confusion_matrix']['TN']}, FN={results['confusion_matrix']['FN']}")
    print(f"\nMetrics:")
    print(f"  TPR (Recall): {results['metrics']['TPR']:.3f}")
    print(f"  FPR: {results['metrics']['FPR']:.3f}")
    print(f"  Precision: {results['metrics']['precision']:.3f}")
    print(f"  F1 Score: {results['metrics']['f1_score']:.3f}")
    print(f"  Accuracy: {results['metrics']['accuracy']:.3f}")


if __name__ == '__main__':
    main()
