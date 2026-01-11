"""
Statistical Analysis Module for Neural Pulse

Computes comprehensive statistics on entropy and attention signals to:
1. Identify discriminative features between attack and normal traces
2. Derive optimal threshold values for STL formulas
3. Generate ROC curves for threshold selection
4. Perform statistical significance tests

Key outputs:
- Distribution statistics (mean, std, percentiles)
- Temporal features (max, min, duration, variance)
- ROC curves and AUC scores
- Optimal threshold recommendations
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalStatistics:
    """Statistics for a signal (entropy or attention)"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentile_25: float
    percentile_75: float
    variance: float
    skewness: float
    kurtosis: float


@dataclass
class TemporalFeatures:
    """Temporal characteristics of a trace"""
    max_value: float
    min_value: float
    max_position: int  # Token position of maximum
    min_position: int  # Token position of minimum
    duration_above_threshold: int  # Tokens where signal > threshold
    sustained_high_count: int  # Count of sustained high periods (3+ consecutive)
    variance: float
    trend_slope: float  # Linear regression slope


@dataclass
class ThresholdAnalysis:
    """Analysis results for a specific threshold"""
    threshold: float
    tpr: float  # True positive rate (sensitivity)
    fpr: float  # False positive rate
    tnr: float  # True negative rate (specificity)
    fnr: float  # False negative rate
    precision: float
    recall: float
    f1_score: float
    accuracy: float


class SignalAnalyzer:
    """
    Statistical analyzer for entropy and attention signals.

    Computes comprehensive statistics and identifies optimal thresholds
    for distinguishing SECA attacks from normal prompts.
    """

    def __init__(self):
        """Initialize signal analyzer."""
        logger.info("SignalAnalyzer initialized")

    def compute_signal_statistics(self, signal_values: List[float]) -> SignalStatistics:
        """
        Compute comprehensive statistics for a signal.

        Args:
            signal_values: List of signal values

        Returns:
            SignalStatistics object with computed metrics
        """
        if len(signal_values) == 0:
            return SignalStatistics(
                mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
                percentile_25=0.0, percentile_75=0.0,
                variance=0.0, skewness=0.0, kurtosis=0.0
            )

        arr = np.array(signal_values)

        return SignalStatistics(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            percentile_25=float(np.percentile(arr, 25)),
            percentile_75=float(np.percentile(arr, 75)),
            variance=float(np.var(arr)),
            skewness=float(stats.skew(arr)) if len(arr) > 1 else 0.0,
            kurtosis=float(stats.kurtosis(arr)) if len(arr) > 1 else 0.0
        )

    def compute_temporal_features(
        self,
        signal_trace: List[float],
        threshold: Optional[float] = None
    ) -> TemporalFeatures:
        """
        Compute temporal features from a signal trace.

        Args:
            signal_trace: Time series of signal values
            threshold: Optional threshold for duration computation

        Returns:
            TemporalFeatures object
        """
        if len(signal_trace) == 0:
            return TemporalFeatures(
                max_value=0.0, min_value=0.0,
                max_position=0, min_position=0,
                duration_above_threshold=0,
                sustained_high_count=0,
                variance=0.0, trend_slope=0.0
            )

        arr = np.array(signal_trace)

        # Basic extrema
        max_value = float(np.max(arr))
        min_value = float(np.min(arr))
        max_position = int(np.argmax(arr))
        min_position = int(np.argmin(arr))

        # Threshold-based features
        if threshold is not None:
            above_threshold = arr > threshold
            duration_above_threshold = int(np.sum(above_threshold))

            # Count sustained high periods (3+ consecutive above threshold)
            sustained_high_count = 0
            current_streak = 0
            for val in above_threshold:
                if val:
                    current_streak += 1
                    if current_streak == 3:
                        sustained_high_count += 1
                else:
                    current_streak = 0
        else:
            duration_above_threshold = 0
            sustained_high_count = 0

        # Trend analysis (linear regression)
        if len(arr) > 1:
            x = np.arange(len(arr))
            slope, _ = np.polyfit(x, arr, 1)
            trend_slope = float(slope)
        else:
            trend_slope = 0.0

        return TemporalFeatures(
            max_value=max_value,
            min_value=min_value,
            max_position=max_position,
            min_position=min_position,
            duration_above_threshold=duration_above_threshold,
            sustained_high_count=sustained_high_count,
            variance=float(np.var(arr)),
            trend_slope=trend_slope
        )

    def compare_distributions(
        self,
        attack_values: List[float],
        normal_values: List[float]
    ) -> Dict[str, float]:
        """
        Compare two distributions using statistical tests.

        Args:
            attack_values: Signal values from attack traces
            normal_values: Signal values from normal traces

        Returns:
            Dict with test results
        """
        if len(attack_values) == 0 or len(normal_values) == 0:
            return {
                't_statistic': 0.0,
                't_pvalue': 1.0,
                'ks_statistic': 0.0,
                'ks_pvalue': 1.0,
                'effect_size_cohens_d': 0.0
            }

        attack_arr = np.array(attack_values)
        normal_arr = np.array(normal_values)

        # T-test (parametric)
        t_stat, t_pval = stats.ttest_ind(attack_arr, normal_arr)

        # Kolmogorov-Smirnov test (non-parametric)
        ks_stat, ks_pval = stats.ks_2samp(attack_arr, normal_arr)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(attack_arr) - 1) * np.var(attack_arr) +
             (len(normal_arr) - 1) * np.var(normal_arr)) /
            (len(attack_arr) + len(normal_arr) - 2)
        )
        cohens_d = (np.mean(attack_arr) - np.mean(normal_arr)) / pooled_std if pooled_std > 0 else 0.0

        return {
            't_statistic': float(t_stat),
            't_pvalue': float(t_pval),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'effect_size_cohens_d': float(cohens_d)
        }

    def compute_roc_curve(
        self,
        signal_values: List[float],
        labels: List[bool],
        signal_name: str = 'entropy',
        higher_is_attack: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute ROC curve for threshold selection.

        Args:
            signal_values: Signal values (e.g., mean entropy)
            labels: True if attack, False if normal
            signal_name: Name of signal for logging
            higher_is_attack: If True, higher values indicate attack (entropy)
                             If False, lower values indicate attack (attention)

        Returns:
            Tuple of (fpr_array, tpr_array, auc_score)
        """
        if len(signal_values) == 0 or len(labels) == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

        # Convert labels to binary (1 = attack, 0 = normal)
        y_true = np.array([1 if label else 0 for label in labels])
        y_scores = np.array(signal_values)

        # For attention (lower is attack), negate scores
        if not higher_is_attack:
            y_scores = -y_scores

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)

        logger.info(f"ROC curve for {signal_name}: AUC = {auc_score:.3f}")

        return fpr, tpr, auc_score

    def find_optimal_threshold(
        self,
        signal_values: List[float],
        labels: List[bool],
        metric: str = 'f1',
        higher_is_attack: bool = True
    ) -> Tuple[float, ThresholdAnalysis]:
        """
        Find optimal threshold by maximizing a metric.

        Args:
            signal_values: Signal values
            labels: True if attack, False if normal
            metric: 'f1', 'accuracy', or 'youden' (TPR - FPR)
            higher_is_attack: If True, values > threshold are attacks

        Returns:
            Tuple of (optimal_threshold, analysis_at_threshold)
        """
        if len(signal_values) == 0 or len(labels) == 0:
            return 0.0, ThresholdAnalysis(
                threshold=0.0, tpr=0.0, fpr=0.0, tnr=0.0, fnr=0.0,
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0
            )

        y_true = np.array([1 if label else 0 for label in labels])
        y_scores = np.array(signal_values)

        # For attention (lower is attack), negate
        if not higher_is_attack:
            y_scores = -y_scores

        # Get precision-recall curve for threshold candidates
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Compute F1 scores for each threshold
        best_score = -1.0
        best_threshold = 0.0
        best_analysis = None

        # Also try percentile-based thresholds
        candidate_thresholds = np.percentile(y_scores, np.arange(10, 91, 5))

        for threshold in candidate_thresholds:
            # y_scores already negated if not higher_is_attack, so always use >= for prediction
            y_pred = (y_scores >= threshold).astype(int)

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            # Compute metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tpr
            f1 = 2 * precision_val * recall_val / (precision_val + recall_val) \
                if (precision_val + recall_val) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Select based on metric
            if metric == 'f1':
                score = f1
            elif metric == 'accuracy':
                score = accuracy
            elif metric == 'youden':
                score = tpr - fpr  # Youden's J statistic
            else:
                score = f1

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_analysis = ThresholdAnalysis(
                    threshold=float(threshold),
                    tpr=float(tpr),
                    fpr=float(fpr),
                    tnr=float(tnr),
                    fnr=float(fnr),
                    precision=float(precision_val),
                    recall=float(recall_val),
                    f1_score=float(f1),
                    accuracy=float(accuracy)
                )

        # Convert back if negated
        if not higher_is_attack:
            best_threshold = -best_threshold

        logger.info(f"Optimal threshold: {best_threshold:.3f} (metric={metric}, score={best_score:.3f})")

        return best_threshold, best_analysis

    def analyze_dataset(
        self,
        traces: List[Dict],
        validations: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Perform comprehensive statistical analysis on a dataset.

        Args:
            traces: List of trace dicts with entropy/attention signals
            validations: List of validation dicts with is_hallucination labels
            output_path: Optional path to save results JSON

        Returns:
            Dict with complete analysis results
        """
        logger.info(f"Analyzing {len(traces)} traces...")

        # Split by label
        attack_entropy = []
        attack_attention = []
        attack_perplexity = []
        attack_attention_entropy = []
        normal_entropy = []
        normal_attention = []
        normal_perplexity = []
        normal_attention_entropy = []

        for trace, validation in zip(traces, validations):
            entropy_trace = trace.get('entropy_trace', [])
            attention_trace = trace.get('attention_trace', [])
            perplexity_trace = trace.get('perplexity_trace', [])
            attention_entropy_trace = trace.get('attention_entropy_trace', [])

            if len(entropy_trace) == 0:
                continue

            mean_entropy = np.mean(entropy_trace)
            mean_attention = np.mean(attention_trace) if len(attention_trace) > 0 else 0.0
            mean_perplexity = np.mean(perplexity_trace) if len(perplexity_trace) > 0 else 0.0
            mean_attention_entropy = np.mean(attention_entropy_trace) if len(attention_entropy_trace) > 0 else 0.0

            if validation['is_hallucination']:
                attack_entropy.append(mean_entropy)
                attack_attention.append(mean_attention)
                attack_perplexity.append(mean_perplexity)
                attack_attention_entropy.append(mean_attention_entropy)
            else:
                normal_entropy.append(mean_entropy)
                normal_attention.append(mean_attention)
                normal_perplexity.append(mean_perplexity)
                normal_attention_entropy.append(mean_attention_entropy)

        # Compute statistics
        results = {
            'dataset_summary': {
                'total_traces': len(traces),
                'attack_traces': len(attack_entropy),
                'normal_traces': len(normal_entropy)
            },
            'entropy': {
                'attack_stats': asdict(self.compute_signal_statistics(attack_entropy)),
                'normal_stats': asdict(self.compute_signal_statistics(normal_entropy)),
                'comparison': self.compare_distributions(attack_entropy, normal_entropy)
            },
            'attention': {
                'attack_stats': asdict(self.compute_signal_statistics(attack_attention)),
                'normal_stats': asdict(self.compute_signal_statistics(normal_attention)),
                'comparison': self.compare_distributions(attack_attention, normal_attention)
            }
        }

        # Add new signals if they exist
        if len(attack_perplexity) > 0 and any(p > 0 for p in attack_perplexity):
            results['perplexity'] = {
                'attack_stats': asdict(self.compute_signal_statistics(attack_perplexity)),
                'normal_stats': asdict(self.compute_signal_statistics(normal_perplexity)),
                'comparison': self.compare_distributions(attack_perplexity, normal_perplexity)
            }

        if len(attack_attention_entropy) > 0 and any(a > 0 for a in attack_attention_entropy):
            results['attention_entropy'] = {
                'attack_stats': asdict(self.compute_signal_statistics(attack_attention_entropy)),
                'normal_stats': asdict(self.compute_signal_statistics(normal_attention_entropy)),
                'comparison': self.compare_distributions(attack_attention_entropy, normal_attention_entropy)
            }

        # ROC curves
        all_entropy = attack_entropy + normal_entropy
        all_attention = attack_attention + normal_attention
        all_perplexity = attack_perplexity + normal_perplexity
        all_attention_entropy = attack_attention_entropy + normal_attention_entropy
        all_labels = [True] * len(attack_entropy) + [False] * len(normal_entropy)

        fpr_entropy, tpr_entropy, auc_entropy = self.compute_roc_curve(
            all_entropy, all_labels, 'entropy', higher_is_attack=True
        )
        fpr_attention, tpr_attention, auc_attention = self.compute_roc_curve(
            all_attention, all_labels, 'attention', higher_is_attack=False
        )

        results['roc_curves'] = {
            'entropy': {
                'fpr': fpr_entropy.tolist(),
                'tpr': tpr_entropy.tolist(),
                'auc': float(auc_entropy)
            },
            'attention': {
                'fpr': fpr_attention.tolist(),
                'tpr': tpr_attention.tolist(),
                'auc': float(auc_attention)
            }
        }

        # Add ROC for new signals if they exist
        if len(all_perplexity) > 0 and any(p > 0 for p in all_perplexity):
            fpr_perp, tpr_perp, auc_perp = self.compute_roc_curve(
                all_perplexity, all_labels, 'perplexity', higher_is_attack=True
            )
            results['roc_curves']['perplexity'] = {
                'fpr': fpr_perp.tolist(),
                'tpr': tpr_perp.tolist(),
                'auc': float(auc_perp)
            }

        if len(all_attention_entropy) > 0 and any(a > 0 for a in all_attention_entropy):
            fpr_attn_ent, tpr_attn_ent, auc_attn_ent = self.compute_roc_curve(
                all_attention_entropy, all_labels, 'attention_entropy', higher_is_attack=True
            )
            results['roc_curves']['attention_entropy'] = {
                'fpr': fpr_attn_ent.tolist(),
                'tpr': tpr_attn_ent.tolist(),
                'auc': float(auc_attn_ent)
            }

        # Optimal thresholds
        threshold_entropy, analysis_entropy = self.find_optimal_threshold(
            all_entropy, all_labels, metric='f1', higher_is_attack=True
        )
        threshold_attention, analysis_attention = self.find_optimal_threshold(
            all_attention, all_labels, metric='f1', higher_is_attack=False
        )

        results['optimal_thresholds'] = {
            'entropy': {
                'threshold': threshold_entropy,
                'analysis': asdict(analysis_entropy)
            },
            'attention': {
                'threshold': threshold_attention,
                'analysis': asdict(analysis_attention)
            }
        }

        # Add optimal thresholds for new signals if they exist
        if len(all_perplexity) > 0 and any(p > 0 for p in all_perplexity):
            threshold_perp, analysis_perp = self.find_optimal_threshold(
                all_perplexity, all_labels, metric='f1', higher_is_attack=True
            )
            results['optimal_thresholds']['perplexity'] = {
                'threshold': threshold_perp,
                'analysis': asdict(analysis_perp)
            }

        if len(all_attention_entropy) > 0 and any(a > 0 for a in all_attention_entropy):
            threshold_attn_ent, analysis_attn_ent = self.find_optimal_threshold(
                all_attention_entropy, all_labels, metric='f1', higher_is_attack=True
            )
            results['optimal_thresholds']['attention_entropy'] = {
                'threshold': threshold_attn_ent,
                'analysis': asdict(analysis_attn_ent)
            }

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Analysis results saved to: {output_path}")

        return results


def main():
    """CLI interface for statistical analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Neural Pulse Statistical Analysis')
    parser.add_argument('--traces', type=str, required=True, help='Path to traces JSON')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation JSON')
    parser.add_argument('--output', type=str, required=True, help='Path to output analysis JSON')

    args = parser.parse_args()

    # Load data
    with open(args.traces, 'r') as f:
        traces_data = json.load(f)
        traces = traces_data.get('traces', traces_data)

    with open(args.validation, 'r') as f:
        validation_data = json.load(f)
        validations = validation_data.get('validation_results', validation_data)

    # Run analysis
    analyzer = SignalAnalyzer()
    results = analyzer.analyze_dataset(traces, validations, output_path=args.output)

    # Print summary
    print("\n=== Statistical Analysis Summary ===")
    print(f"Total traces: {results['dataset_summary']['total_traces']}")
    print(f"Attack traces: {results['dataset_summary']['attack_traces']}")
    print(f"Normal traces: {results['dataset_summary']['normal_traces']}")
    print("\nEntropy:")
    print(f"  Attack mean: {results['entropy']['attack_stats']['mean']:.3f}")
    print(f"  Normal mean: {results['entropy']['normal_stats']['mean']:.3f}")
    print(f"  ROC AUC: {results['roc_curves']['entropy']['auc']:.3f}")
    print(f"  Optimal threshold: {results['optimal_thresholds']['entropy']['threshold']:.3f}")
    print(f"  F1 score: {results['optimal_thresholds']['entropy']['analysis']['f1_score']:.3f}")
    print("\nAttention:")
    print(f"  Attack mean: {results['attention']['attack_stats']['mean']:.3f}")
    print(f"  Normal mean: {results['attention']['normal_stats']['mean']:.3f}")
    print(f"  ROC AUC: {results['roc_curves']['attention']['auc']:.3f}")
    print(f"  Optimal threshold: {results['optimal_thresholds']['attention']['threshold']:.3f}")
    print(f"  F1 score: {results['optimal_thresholds']['attention']['analysis']['f1_score']:.3f}")


if __name__ == '__main__':
    main()
