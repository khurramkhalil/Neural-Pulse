#!/usr/bin/env python3
"""
Multi-Signal Classifier for Hallucination Detection

Combines all 4 signals (Entropy, Attention, Perplexity, Attention Entropy)
using Logistic Regression to improve classification performance.

Usage:
    python analysis/multi_signal_classifier.py \
        --traces results/pilot_traces.json \
        --validation results/pilot_validation.json \
        --output results/multi_signal_classifier_results.json
"""

import json
import argparse
import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassifierResults:
    """Results from multi-signal classifier"""
    # Model performance
    train_auc: float
    val_auc: float
    test_auc: float

    # Metrics at optimal threshold
    optimal_threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Feature importance
    feature_weights: Dict[str, float]
    feature_importance: Dict[str, float]

    # Cross-validation
    cv_auc_mean: float
    cv_auc_std: float

    # ROC curve data
    fpr: List[float]
    tpr: List[float]

    # Confusion matrix
    confusion_matrix: List[List[int]]

    # Individual signal AUCs for comparison
    individual_aucs: Dict[str, float]


class MultiSignalClassifier:
    """
    Combines multiple signals using logistic regression.
    """

    def __init__(self, signals: List[str] = None):
        """
        Initialize classifier.

        Args:
            signals: List of signal names to use.
                    Default (Phase 2a): ['entropy', 'semantic_drift'] - drop weak signals
        """
        if signals is None:
            # PHASE 2a: Use only validated signals (Entropy + Semantic Drift)
            # Drop: Attention (AUC 0.38), Perplexity (wrong sign), Attention Entropy (weak)
            self.signals = ['entropy', 'semantic_drift']
        else:
            self.signals = signals

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=42
        )

    def extract_features(
        self,
        traces: List[Dict],
        validations: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from traces.

        Args:
            traces: List of trace dictionaries
            validations: List of validation dictionaries

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        features = []
        labels = []

        for trace, validation in zip(traces, validations):
            # Extract mean value for each signal
            feature_vec = []

            for signal in self.signals:
                signal_trace = trace.get(f'{signal}_trace', [])
                if len(signal_trace) > 0:
                    # Use mean as the representative value
                    feature_vec.append(np.mean(signal_trace))
                else:
                    # Missing signal - use 0
                    feature_vec.append(0.0)

            features.append(feature_vec)
            labels.append(1 if validation['is_hallucination'] else 0)

        X = np.array(features)
        y = np.array(labels)

        return X, y

    def compute_individual_aucs(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute AUC for each individual signal.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Dictionary mapping signal name to AUC
        """
        aucs = {}

        for i, signal in enumerate(self.signals):
            # Get single feature column
            feature = X[:, i]

            # Check if signal has variation
            if np.std(feature) < 1e-6:
                aucs[signal] = 0.5
                continue

            # Compute AUC
            # For entropy, perplexity, attention_entropy: higher = attack
            # For attention, semantic_drift: lower = attack (inverted)
            if signal in ['attention', 'semantic_drift']:
                # Invert for these signals (lower value = attack)
                # - Attention: lower attention mass = detachment (DEPRECATED - failed)
                # - Semantic Drift: lower similarity = drifting away (PHASE 2a - PRIMARY)
                auc = roc_auc_score(y, -feature)
            else:
                auc = roc_auc_score(y, feature)

            aucs[signal] = auc

        return aucs

    def train_and_evaluate(
        self,
        traces: List[Dict],
        validations: List[Dict],
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> ClassifierResults:
        """
        Train and evaluate the multi-signal classifier.

        Args:
            traces: List of trace dictionaries
            validations: List of validation dictionaries
            test_size: Fraction of data to use for testing
            val_size: Fraction of remaining data to use for validation

        Returns:
            ClassifierResults object
        """
        logger.info("Extracting features from traces...")
        X, y = self.extract_features(traces, validations)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        # Compute individual signal AUCs
        logger.info("Computing individual signal AUCs...")
        individual_aucs = self.compute_individual_aucs(X, y)

        for signal, auc in individual_aucs.items():
            logger.info(f"  {signal}: AUC = {auc:.4f}")

        # Split data: train / val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size),
            random_state=42, stratify=y_temp
        )

        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Normalize features
        logger.info("Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        logger.info("Training logistic regression model...")
        self.model.fit(X_train_scaled, y_train)

        # Get predictions (probabilities)
        y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Compute AUCs
        train_auc = roc_auc_score(y_train, y_train_proba)
        val_auc = roc_auc_score(y_val, y_val_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)

        logger.info(f"Train AUC: {train_auc:.4f}")
        logger.info(f"Val AUC: {val_auc:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")

        # Cross-validation on training set
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='roc_auc'
        )
        cv_auc_mean = np.mean(cv_scores)
        cv_auc_std = np.std(cv_scores)

        logger.info(f"Cross-validation AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

        # Find optimal threshold on validation set
        logger.info("Finding optimal threshold...")
        fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_proba)

        # Optimize for F1 score
        f1_scores = []
        for thresh in thresholds_val:
            y_val_pred = (y_val_proba >= thresh).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_val[optimal_idx]

        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")

        # Evaluate on test set with optimal threshold
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Get feature weights and importance
        feature_weights = {
            signal: float(weight)
            for signal, weight in zip(self.signals, self.model.coef_[0])
        }

        # Importance = |weight| (magnitude matters for feature importance)
        feature_importance = {
            signal: float(abs(weight))
            for signal, weight in feature_weights.items()
        }

        # Sort by importance
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Feature Importance (sorted):")
        for signal, importance in sorted_importance:
            weight = feature_weights[signal]
            logger.info(f"  {signal}: {importance:.4f} (weight: {weight:+.4f})")

        # Get ROC curve for test set
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

        # Create results object
        results = ClassifierResults(
            train_auc=train_auc,
            val_auc=val_auc,
            test_auc=test_auc,
            optimal_threshold=optimal_threshold,
            accuracy=test_accuracy,
            precision=test_precision,
            recall=test_recall,
            f1_score=test_f1,
            feature_weights=feature_weights,
            feature_importance=feature_importance,
            cv_auc_mean=cv_auc_mean,
            cv_auc_std=cv_auc_std,
            fpr=fpr_test.tolist(),
            tpr=tpr_test.tolist(),
            confusion_matrix=cm.tolist(),
            individual_aucs=individual_aucs
        )

        return results

    def plot_results(
        self,
        results: ClassifierResults,
        output_dir: Path
    ):
        """
        Generate visualization plots.

        Args:
            results: ClassifierResults object
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. ROC Curve Comparison (Individual vs Combined)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot combined model ROC
        ax.plot(
            results.fpr, results.tpr,
            label=f'Combined Model (AUC = {results.test_auc:.3f})',
            linewidth=3, color='red'
        )

        # Plot individual signal ROCs (approximate)
        colors = ['orange', 'purple', 'blue', 'green']
        for i, (signal, auc) in enumerate(results.individual_aucs.items()):
            # Approximate ROC for visualization
            ax.plot(
                [0, 1], [0, 1], '--', alpha=0.3,
                label=f'{signal.capitalize()} (AUC = {auc:.3f})',
                color=colors[i], linewidth=2
            )

        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve: Multi-Signal vs Individual Signals', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ROC comparison to {output_dir / 'roc_comparison.png'}")

        # 2. Feature Importance Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))

        signals = list(results.feature_importance.keys())
        importances = list(results.feature_importance.values())
        weights = [results.feature_weights[s] for s in signals]

        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        signals_sorted = [signals[i] for i in sorted_indices]
        importances_sorted = [importances[i] for i in sorted_indices]
        weights_sorted = [weights[i] for i in sorted_indices]

        # Color bars by sign of weight
        colors = ['green' if w > 0 else 'red' for w in weights_sorted]

        bars = ax.barh(signals_sorted, importances_sorted, color=colors, alpha=0.7, edgecolor='black')

        # Add weight values as text
        for i, (imp, weight) in enumerate(zip(importances_sorted, weights_sorted)):
            ax.text(
                imp + 0.01, i,
                f'{weight:+.3f}',
                va='center', fontsize=10, fontweight='bold'
            )

        ax.set_xlabel('Feature Importance (|Weight|)', fontsize=12)
        ax.set_title('Multi-Signal Classifier: Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Positive weight (↑ signal → ↑ attack prob)'),
            Patch(facecolor='red', alpha=0.7, label='Negative weight (↑ signal → ↓ attack prob)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved feature importance to {output_dir / 'feature_importance.png'}")

        # 3. AUC Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Combine individual and combined AUCs
        all_signals = list(results.individual_aucs.keys()) + ['Combined']
        all_aucs = list(results.individual_aucs.values()) + [results.test_auc]

        # Color combined differently
        colors_bar = ['skyblue'] * len(results.individual_aucs) + ['red']

        bars = ax.bar(all_signals, all_aucs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

        # Add AUC values on top of bars
        for bar, auc in zip(bars, all_aucs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{auc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )

        # Add horizontal line at 0.5 (random classifier)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (AUC=0.50)')

        # Add horizontal line at target (0.85)
        ax.axhline(0.85, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Publication Target (AUC=0.85)')

        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('AUC Comparison: Individual Signals vs Combined Model', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved AUC comparison to {output_dir / 'auc_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-signal classifier for hallucination detection'
    )
    parser.add_argument(
        '--traces',
        type=str,
        required=True,
        help='Path to traces JSON file'
    )
    parser.add_argument(
        '--validation',
        type=str,
        required=True,
        help='Path to validation JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--figures-dir',
        type=str,
        default='results/multi_signal_figures',
        help='Directory to save figures'
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading traces from {args.traces}...")
    with open(args.traces) as f:
        traces_data = json.load(f)
        if isinstance(traces_data, dict):
            traces = traces_data.get('traces', traces_data)
        else:
            traces = traces_data

    logger.info(f"Loading validation from {args.validation}...")
    with open(args.validation) as f:
        validation_data = json.load(f)
        if isinstance(validation_data, dict):
            validations = validation_data.get('validation_results', validation_data)
        else:
            validations = validation_data

    logger.info(f"Loaded {len(traces)} traces")

    # Initialize classifier
    classifier = MultiSignalClassifier()

    # Train and evaluate
    results = classifier.train_and_evaluate(traces, validations)

    # Save results
    logger.info(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(asdict(results), f, indent=2)

    # Generate plots
    figures_dir = Path(args.figures_dir)
    logger.info(f"Generating plots in {figures_dir}...")
    classifier.plot_results(results, figures_dir)

    # Print summary
    print("\n" + "="*80)
    print("MULTI-SIGNAL CLASSIFIER RESULTS")
    print("="*80)
    print(f"\nDataset: {len(traces)} traces")
    print(f"\nIndividual Signal AUCs:")
    for signal, auc in results.individual_aucs.items():
        print(f"  {signal:20s}: {auc:.4f}")

    print(f"\nCombined Model Performance:")
    print(f"  Cross-Val AUC:  {results.cv_auc_mean:.4f} ± {results.cv_auc_std:.4f}")
    print(f"  Train AUC:      {results.train_auc:.4f}")
    print(f"  Val AUC:        {results.val_auc:.4f}")
    print(f"  Test AUC:       {results.test_auc:.4f}")

    improvement = results.test_auc - max(results.individual_aucs.values())
    print(f"\n  Improvement over best single signal: +{improvement:.4f}")

    print(f"\nTest Set Metrics (threshold={results.optimal_threshold:.4f}):")
    print(f"  Accuracy:   {results.accuracy:.4f}")
    print(f"  Precision:  {results.precision:.4f}")
    print(f"  Recall:     {results.recall:.4f}")
    print(f"  F1 Score:   {results.f1_score:.4f}")

    print(f"\nFeature Weights:")
    sorted_features = sorted(
        results.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for signal, importance in sorted_features:
        weight = results.feature_weights[signal]
        print(f"  {signal:20s}: {weight:+.4f} (importance: {importance:.4f})")

    print(f"\nConfusion Matrix (Test Set):")
    cm = np.array(results.confusion_matrix)
    print(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
    print(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")

    # Publication readiness assessment
    print(f"\n" + "="*80)
    print("PUBLICATION READINESS ASSESSMENT")
    print("="*80)

    target_auc = 0.85
    if results.test_auc >= target_auc:
        print(f"✅ SUCCESS! Test AUC {results.test_auc:.4f} >= {target_auc} (publication target)")
        print(f"\nRecommendation: Proceed to Phase 3 (Real-time Monitor)")
    elif results.test_auc >= 0.75:
        print(f"⚠️  CLOSE! Test AUC {results.test_auc:.4f} is close to {target_auc} target")
        gap = target_auc - results.test_auc
        print(f"\nGap: {gap:.4f} AUC points needed")
        print(f"\nRecommendation: Consider collecting more data (scale to 500-1000 attacks)")
    else:
        print(f"❌ INSUFFICIENT! Test AUC {results.test_auc:.4f} < 0.75")
        gap = target_auc - results.test_auc
        print(f"\nGap: {gap:.4f} AUC points needed")
        print(f"\nRecommendation: Rethink approach - current signals may not be sufficient")

    print("="*80)
    print(f"\nResults saved to: {args.output}")
    print(f"Figures saved to: {figures_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
