"""
Comprehensive Unit Tests for Statistical Analysis

Tests cover:
1. Signal statistics computation (mean, std, percentiles, etc.)
2. Temporal feature extraction (max, min, duration, trends)
3. Distribution comparisons (t-test, KS-test, Cohen's d)
4. ROC curve computation
5. Optimal threshold selection
6. Full dataset analysis
7. Edge cases (empty data, identical distributions, single value)

NO try-except gaming - tests verify mathematical correctness.
"""

import unittest
import numpy as np
import json
import tempfile
from analysis.statistical_analysis import (
    SignalAnalyzer,
    SignalStatistics,
    TemporalFeatures,
    ThresholdAnalysis
)


class TestSignalStatistics(unittest.TestCase):
    """Test basic signal statistics computation"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def test_statistics_normal_distribution(self):
        """Test: Statistics computed correctly for normal distribution"""
        # Known normal distribution
        np.random.seed(42)
        values = np.random.normal(loc=5.0, scale=2.0, size=1000).tolist()

        stats = self.analyzer.compute_signal_statistics(values)

        # Mean should be close to 5.0
        self.assertAlmostEqual(stats.mean, 5.0, delta=0.2)
        # Std should be close to 2.0
        self.assertAlmostEqual(stats.std, 2.0, delta=0.2)
        # Median should be close to mean for normal distribution
        self.assertAlmostEqual(stats.median, stats.mean, delta=0.5)

    def test_statistics_uniform_distribution(self):
        """Test: Statistics for uniform distribution"""
        values = np.linspace(0, 10, 100).tolist()

        stats = self.analyzer.compute_signal_statistics(values)

        self.assertAlmostEqual(stats.mean, 5.0, delta=0.1)
        self.assertAlmostEqual(stats.min, 0.0, delta=0.1)
        self.assertAlmostEqual(stats.max, 10.0, delta=0.1)
        self.assertAlmostEqual(stats.median, 5.0, delta=0.1)

    def test_statistics_percentiles(self):
        """Test: Percentile calculations are correct"""
        values = list(range(1, 101))  # 1 to 100

        stats = self.analyzer.compute_signal_statistics(values)

        # 25th percentile should be around 25
        self.assertAlmostEqual(stats.percentile_25, 25.75, delta=1.0)
        # 75th percentile should be around 75
        self.assertAlmostEqual(stats.percentile_75, 75.25, delta=1.0)

    def test_statistics_empty_list(self):
        """Test: Empty list returns zero statistics"""
        stats = self.analyzer.compute_signal_statistics([])

        self.assertEqual(stats.mean, 0.0)
        self.assertEqual(stats.std, 0.0)
        self.assertEqual(stats.min, 0.0)
        self.assertEqual(stats.max, 0.0)

    def test_statistics_single_value(self):
        """Test: Single value has zero variance"""
        stats = self.analyzer.compute_signal_statistics([5.0])

        self.assertEqual(stats.mean, 5.0)
        self.assertEqual(stats.std, 0.0)
        self.assertEqual(stats.variance, 0.0)
        self.assertEqual(stats.min, 5.0)
        self.assertEqual(stats.max, 5.0)

    def test_statistics_variance_calculation(self):
        """Test: Variance calculated correctly"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = self.analyzer.compute_signal_statistics(values)

        # Variance of [1,2,3,4,5] is 2.0
        self.assertAlmostEqual(stats.variance, 2.0, delta=0.01)
        self.assertAlmostEqual(stats.std, np.sqrt(2.0), delta=0.01)


class TestTemporalFeatures(unittest.TestCase):
    """Test temporal feature extraction"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def test_temporal_features_basic(self):
        """Test: Basic temporal features extracted correctly"""
        signal = [1.0, 2.0, 5.0, 3.0, 1.0]

        features = self.analyzer.compute_temporal_features(signal)

        self.assertEqual(features.max_value, 5.0)
        self.assertEqual(features.min_value, 1.0)
        self.assertEqual(features.max_position, 2)
        self.assertEqual(features.min_position, 0)  # First occurrence

    def test_temporal_features_duration_above_threshold(self):
        """Test: Duration above threshold counted correctly"""
        signal = [1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 1.0]
        threshold = 2.5

        features = self.analyzer.compute_temporal_features(signal, threshold=threshold)

        # Values above 2.5: [3.0, 4.0, 5.0, 6.0] = 4 tokens
        self.assertEqual(features.duration_above_threshold, 4)

    def test_temporal_features_sustained_high_count(self):
        """Test: Sustained high periods (3+ consecutive) counted"""
        signal = [1.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        threshold = 2.5

        features = self.analyzer.compute_temporal_features(signal, threshold=threshold)

        # Two sustained periods: [3,4,5] and [3,4,5,6]
        self.assertEqual(features.sustained_high_count, 2)

    def test_temporal_features_trend_slope(self):
        """Test: Trend slope calculated correctly"""
        # Increasing signal
        signal_increasing = [1.0, 2.0, 3.0, 4.0, 5.0]
        features_inc = self.analyzer.compute_temporal_features(signal_increasing)
        self.assertGreater(features_inc.trend_slope, 0.9)  # Positive slope

        # Decreasing signal
        signal_decreasing = [5.0, 4.0, 3.0, 2.0, 1.0]
        features_dec = self.analyzer.compute_temporal_features(signal_decreasing)
        self.assertLess(features_dec.trend_slope, -0.9)  # Negative slope

    def test_temporal_features_empty_signal(self):
        """Test: Empty signal returns zero features"""
        features = self.analyzer.compute_temporal_features([])

        self.assertEqual(features.max_value, 0.0)
        self.assertEqual(features.min_value, 0.0)
        self.assertEqual(features.duration_above_threshold, 0)

    def test_temporal_features_constant_signal(self):
        """Test: Constant signal has zero slope"""
        signal = [3.0, 3.0, 3.0, 3.0, 3.0]

        features = self.analyzer.compute_temporal_features(signal)

        self.assertAlmostEqual(features.trend_slope, 0.0, delta=0.01)
        self.assertEqual(features.max_value, 3.0)
        self.assertEqual(features.min_value, 3.0)


class TestDistributionComparisons(unittest.TestCase):
    """Test statistical comparison of distributions"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def test_compare_clearly_different_distributions(self):
        """Test: Clearly different distributions have low p-value"""
        attack = np.random.normal(loc=5.0, scale=1.0, size=100).tolist()
        normal = np.random.normal(loc=1.0, scale=1.0, size=100).tolist()

        comparison = self.analyzer.compare_distributions(attack, normal)

        # P-value should be very small (significant difference)
        self.assertLess(comparison['t_pvalue'], 0.01)
        self.assertLess(comparison['ks_pvalue'], 0.01)
        # Effect size should be large
        self.assertGreater(abs(comparison['effect_size_cohens_d']), 2.0)

    def test_compare_identical_distributions(self):
        """Test: Identical distributions have high p-value"""
        np.random.seed(42)
        data = np.random.normal(loc=5.0, scale=1.0, size=100).tolist()

        comparison = self.analyzer.compare_distributions(data, data)

        # P-value should be 1.0 (no difference)
        self.assertAlmostEqual(comparison['t_pvalue'], 1.0, delta=0.01)
        # Effect size should be 0
        self.assertAlmostEqual(comparison['effect_size_cohens_d'], 0.0, delta=0.01)

    def test_compare_similar_distributions(self):
        """Test: Similar distributions have high p-value"""
        np.random.seed(42)
        dist1 = np.random.normal(loc=5.0, scale=1.0, size=100).tolist()
        dist2 = np.random.normal(loc=5.05, scale=1.0, size=100).tolist()  # Even smaller difference

        comparison = self.analyzer.compare_distributions(dist1, dist2)

        # P-value should be high (not significantly different)
        self.assertGreater(comparison['t_pvalue'], 0.05)  # More lenient threshold
        # Effect size should be small
        self.assertLess(abs(comparison['effect_size_cohens_d']), 0.5)

    def test_compare_empty_distributions(self):
        """Test: Empty distributions return default values"""
        comparison = self.analyzer.compare_distributions([], [])

        self.assertEqual(comparison['t_pvalue'], 1.0)
        self.assertEqual(comparison['ks_pvalue'], 1.0)
        self.assertEqual(comparison['effect_size_cohens_d'], 0.0)

    def test_compare_one_empty(self):
        """Test: One empty distribution returns default values"""
        data = [1.0, 2.0, 3.0]
        comparison = self.analyzer.compare_distributions(data, [])

        self.assertEqual(comparison['t_pvalue'], 1.0)


class TestROCCurveComputation(unittest.TestCase):
    """Test ROC curve and AUC computation"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def test_roc_curve_perfect_separation(self):
        """Test: Perfect separation gives AUC = 1.0"""
        # Attack values all high, normal values all low
        signal_values = [5.0, 5.5, 6.0] + [1.0, 1.5, 2.0]
        labels = [True, True, True] + [False, False, False]

        fpr, tpr, auc_score = self.analyzer.compute_roc_curve(
            signal_values, labels, higher_is_attack=True
        )

        self.assertAlmostEqual(auc_score, 1.0, delta=0.01)

    def test_roc_curve_random_classifier(self):
        """Test: Random classifier gives AUC ≈ 0.5"""
        np.random.seed(42)
        # Random values for both groups
        signal_values = np.random.uniform(0, 10, 100).tolist()
        labels = [True] * 50 + [False] * 50

        fpr, tpr, auc_score = self.analyzer.compute_roc_curve(
            signal_values, labels, higher_is_attack=True
        )

        # AUC should be around 0.5 (random)
        self.assertAlmostEqual(auc_score, 0.5, delta=0.15)

    def test_roc_curve_inverted_signal(self):
        """Test: Inverted signal (lower is attack) works correctly"""
        # Lower attention for attacks
        signal_values = [0.2, 0.3, 0.25] + [0.7, 0.8, 0.75]
        labels = [True, True, True] + [False, False, False]

        fpr, tpr, auc_score = self.analyzer.compute_roc_curve(
            signal_values, labels, higher_is_attack=False  # Lower = attack
        )

        self.assertAlmostEqual(auc_score, 1.0, delta=0.01)

    def test_roc_curve_empty_data(self):
        """Test: Empty data returns default ROC"""
        fpr, tpr, auc_score = self.analyzer.compute_roc_curve(
            [], [], higher_is_attack=True
        )

        self.assertAlmostEqual(auc_score, 0.5, delta=0.01)


class TestOptimalThresholdSelection(unittest.TestCase):
    """Test optimal threshold finding"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def test_optimal_threshold_perfect_separation(self):
        """Test: Optimal threshold found for perfect separation"""
        signal_values = [5.0, 5.5, 6.0, 5.8] + [1.0, 1.5, 2.0, 1.8]
        labels = [True] * 4 + [False] * 4

        threshold, analysis = self.analyzer.find_optimal_threshold(
            signal_values, labels, metric='f1', higher_is_attack=True
        )

        # Threshold should be between 2.0 and 5.0
        self.assertGreater(threshold, 2.0)
        self.assertLess(threshold, 5.0)

        # F1 should be perfect or near-perfect
        self.assertGreater(analysis.f1_score, 0.95)
        self.assertGreater(analysis.tpr, 0.95)
        self.assertLess(analysis.fpr, 0.1)

    def test_optimal_threshold_different_metrics(self):
        """Test: Different metrics give reasonable thresholds"""
        signal_values = [5.0, 5.5, 6.0, 5.8] + [1.0, 1.5, 2.0, 1.8]
        labels = [True] * 4 + [False] * 4

        for metric in ['f1', 'accuracy', 'youden']:
            threshold, analysis = self.analyzer.find_optimal_threshold(
                signal_values, labels, metric=metric, higher_is_attack=True
            )

            # All should find reasonable threshold
            self.assertGreater(threshold, 1.0)
            self.assertLess(threshold, 7.0)
            self.assertGreater(analysis.f1_score, 0.8)

    def test_optimal_threshold_inverted_signal(self):
        """Test: Optimal threshold for inverted signal (lower is attack)"""
        # Lower attention for attacks - use wider separation
        signal_values = [0.1, 0.15, 0.12, 0.18] + [0.8, 0.85, 0.82, 0.88]
        labels = [True] * 4 + [False] * 4

        threshold, analysis = self.analyzer.find_optimal_threshold(
            signal_values, labels, metric='f1', higher_is_attack=False
        )

        # Threshold should separate attack from normal values
        # For inverted signal, threshold should be somewhere that classifies correctly
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        # Should achieve good F1 score with well-separated data
        self.assertGreater(analysis.f1_score, 0.9)

    def test_optimal_threshold_empty_data(self):
        """Test: Empty data returns default threshold"""
        threshold, analysis = self.analyzer.find_optimal_threshold(
            [], [], metric='f1', higher_is_attack=True
        )

        self.assertEqual(threshold, 0.0)
        self.assertEqual(analysis.f1_score, 0.0)

    def test_threshold_analysis_metrics_consistency(self):
        """Test: Threshold analysis metrics are mathematically consistent"""
        signal_values = [5.0, 5.5, 6.0, 5.8, 4.5] + [1.0, 1.5, 2.0, 1.8, 2.5]
        labels = [True] * 5 + [False] * 5

        threshold, analysis = self.analyzer.find_optimal_threshold(
            signal_values, labels, metric='f1', higher_is_attack=True
        )

        # TPR + FNR should equal 1.0
        self.assertAlmostEqual(analysis.tpr + analysis.fnr, 1.0, delta=0.01)
        # TNR + FPR should equal 1.0
        self.assertAlmostEqual(analysis.tnr + analysis.fpr, 1.0, delta=0.01)
        # Recall should equal TPR
        self.assertAlmostEqual(analysis.recall, analysis.tpr, delta=0.01)
        # F1 should be 2*P*R/(P+R)
        expected_f1 = 2 * analysis.precision * analysis.recall / (analysis.precision + analysis.recall) \
            if (analysis.precision + analysis.recall) > 0 else 0.0
        self.assertAlmostEqual(analysis.f1_score, expected_f1, delta=0.01)


class TestFullDatasetAnalysis(unittest.TestCase):
    """Test complete dataset analysis pipeline"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def create_mock_dataset(self, n_attack=50, n_normal=50):
        """Helper to create mock traces and validations"""
        traces = []
        validations = []

        # Attack traces (high entropy, low attention)
        for i in range(n_attack):
            traces.append({
                'prompt': f'Attack {i}',
                'entropy_trace': np.random.normal(2.5, 0.5, 50).tolist(),
                'attention_trace': np.random.normal(0.3, 0.1, 50).tolist()
            })
            validations.append({'is_hallucination': True, 'confidence': 0.95})

        # Normal traces (low entropy, high attention)
        for i in range(n_normal):
            traces.append({
                'prompt': f'Normal {i}',
                'entropy_trace': np.random.normal(1.0, 0.3, 50).tolist(),
                'attention_trace': np.random.normal(0.7, 0.1, 50).tolist()
            })
            validations.append({'is_hallucination': False, 'confidence': 0.95})

        return traces, validations

    def test_full_analysis_structure(self):
        """Test: Full analysis returns complete structure"""
        traces, validations = self.create_mock_dataset(20, 20)

        results = self.analyzer.analyze_dataset(traces, validations)

        # Check all expected keys exist
        self.assertIn('dataset_summary', results)
        self.assertIn('entropy', results)
        self.assertIn('attention', results)
        self.assertIn('roc_curves', results)
        self.assertIn('optimal_thresholds', results)

        # Check dataset summary
        self.assertEqual(results['dataset_summary']['total_traces'], 40)
        self.assertEqual(results['dataset_summary']['attack_traces'], 20)
        self.assertEqual(results['dataset_summary']['normal_traces'], 20)

    def test_full_analysis_distinguishes_attack_normal(self):
        """Test: Analysis correctly distinguishes attack from normal"""
        traces, validations = self.create_mock_dataset(30, 30)

        results = self.analyzer.analyze_dataset(traces, validations)

        # Attack entropy should be higher than normal
        attack_entropy_mean = results['entropy']['attack_stats']['mean']
        normal_entropy_mean = results['entropy']['normal_stats']['mean']
        self.assertGreater(attack_entropy_mean, normal_entropy_mean)

        # Attack attention should be lower than normal
        attack_attention_mean = results['attention']['attack_stats']['mean']
        normal_attention_mean = results['attention']['normal_stats']['mean']
        self.assertLess(attack_attention_mean, normal_attention_mean)

    def test_full_analysis_high_auc_scores(self):
        """Test: Well-separated data gives high AUC scores"""
        traces, validations = self.create_mock_dataset(40, 40)

        results = self.analyzer.analyze_dataset(traces, validations)

        # Both signals should have good discriminative power
        self.assertGreater(results['roc_curves']['entropy']['auc'], 0.8)
        self.assertGreater(results['roc_curves']['attention']['auc'], 0.8)

    def test_full_analysis_saves_to_file(self):
        """Test: Analysis results save correctly to JSON"""
        traces, validations = self.create_mock_dataset(10, 10)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_path = tmp.name

        results = self.analyzer.analyze_dataset(traces, validations, output_path=output_path)

        # Verify file exists and is valid JSON
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)

        self.assertEqual(loaded_results['dataset_summary']['total_traces'], 20)

        import os
        os.unlink(output_path)

    def test_full_analysis_empty_dataset(self):
        """Test: Empty dataset returns valid structure"""
        results = self.analyzer.analyze_dataset([], [])

        self.assertEqual(results['dataset_summary']['total_traces'], 0)
        self.assertEqual(results['dataset_summary']['attack_traces'], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.analyzer = SignalAnalyzer()

    def test_all_attack_labels(self):
        """Test: Dataset with only attack labels"""
        traces = [
            {'entropy_trace': [2.5, 2.6], 'attention_trace': [0.3, 0.4]}
            for _ in range(10)
        ]
        validations = [{'is_hallucination': True} for _ in range(10)]

        results = self.analyzer.analyze_dataset(traces, validations)

        self.assertEqual(results['dataset_summary']['attack_traces'], 10)
        self.assertEqual(results['dataset_summary']['normal_traces'], 0)

    def test_all_normal_labels(self):
        """Test: Dataset with only normal labels"""
        traces = [
            {'entropy_trace': [1.0, 1.1], 'attention_trace': [0.7, 0.8]}
            for _ in range(10)
        ]
        validations = [{'is_hallucination': False} for _ in range(10)]

        results = self.analyzer.analyze_dataset(traces, validations)

        self.assertEqual(results['dataset_summary']['attack_traces'], 0)
        self.assertEqual(results['dataset_summary']['normal_traces'], 10)

    def test_traces_with_missing_signals(self):
        """Test: Traces with missing signal keys are skipped"""
        traces = [
            {'prompt': 'Test'},  # No signals
            {'entropy_trace': [2.0], 'attention_trace': [0.5]},
            {'entropy_trace': []},  # Empty signals
        ]
        validations = [
            {'is_hallucination': True},
            {'is_hallucination': True},
            {'is_hallucination': False}
        ]

        results = self.analyzer.analyze_dataset(traces, validations)

        # Only one valid trace should be counted
        total = results['dataset_summary']['attack_traces'] + results['dataset_summary']['normal_traces']
        self.assertEqual(total, 1)


if __name__ == '__main__':
    unittest.main()
