"""
Comprehensive Unit Tests for STL Formulas

Tests cover:
1. Formula initialization and validation
2. Waffling formula (φ₁) evaluation
3. Detachment formula (φ₂) evaluation
4. Combined formula (φ₃) evaluation
5. Robustness score computation
6. Dataset evaluation
7. Edge cases (empty signals, no violations, perfect violations)

NO try-except gaming - tests verify temporal logic correctness.
"""

import unittest
import numpy as np
from core.stl_formulas import (
    STLFormula,
    STLFormulaParams,
    STLEvaluationResult,
    evaluate_formula_on_dataset
)


class TestFormulaInitialization(unittest.TestCase):
    """Test formula initialization and parameter validation"""

    def test_waffling_formula_requires_theta_H(self):
        """Test: Waffling formula requires theta_H parameter"""
        params = STLFormulaParams(theta_H=None, T=100, w=3)

        with self.assertRaises(ValueError):
            STLFormula('waffling', params)

    def test_detachment_formula_requires_theta_A(self):
        """Test: Detachment formula requires theta_A parameter"""
        params = STLFormulaParams(theta_A=None, T=100, w=3)

        with self.assertRaises(ValueError):
            STLFormula('detachment', params)

    def test_combined_formula_requires_both_thresholds(self):
        """Test: Combined formula requires both theta_H and theta_A"""
        params_missing_H = STLFormulaParams(theta_H=None, theta_A=0.4, T=100, w=3)
        params_missing_A = STLFormulaParams(theta_H=2.0, theta_A=None, T=100, w=3)

        with self.assertRaises(ValueError):
            STLFormula('combined', params_missing_H)

        with self.assertRaises(ValueError):
            STLFormula('combined', params_missing_A)

    def test_valid_formula_initialization(self):
        """Test: Valid parameters create formula successfully"""
        params = STLFormulaParams(theta_H=2.0, theta_A=0.4, T=100, w=3)

        formula_waffling = STLFormula('waffling', params)
        formula_detachment = STLFormula('detachment', params)
        formula_combined = STLFormula('combined', params)

        self.assertEqual(formula_waffling.formula_type, 'waffling')
        self.assertEqual(formula_detachment.formula_type, 'detachment')
        self.assertEqual(formula_combined.formula_type, 'combined')


class TestWafflingFormula(unittest.TestCase):
    """Test φ₁: Eventually Always(H(t) > θ_H)"""

    def setUp(self):
        self.params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        self.formula = STLFormula('waffling', self.params)

    def test_sustained_high_entropy_detected(self):
        """Test: Sustained high entropy (3+ consecutive) triggers violation"""
        # Entropy with sustained high period
        entropy = [1.0, 1.5, 2.5, 2.8, 3.0, 1.5, 1.0]

        result = self.formula.evaluate(entropy_trace=entropy)

        self.assertTrue(result.is_violation)
        self.assertEqual(result.formula_name, 'waffling')
        self.assertIsNotNone(result.violation_time)

    def test_no_sustained_high_entropy(self):
        """Test: Brief spikes don't trigger violation (need w=3 consecutive)"""
        # High entropy but not sustained for 3 tokens
        entropy = [1.0, 2.5, 1.5, 2.8, 1.0]

        result = self.formula.evaluate(entropy_trace=entropy)

        self.assertFalse(result.is_violation)

    def test_all_low_entropy(self):
        """Test: All low entropy doesn't trigger violation"""
        entropy = [1.0, 1.2, 1.1, 0.9, 1.3]

        result = self.formula.evaluate(entropy_trace=entropy)

        self.assertFalse(result.is_violation)

    def test_robustness_score_positive_when_violated(self):
        """Test: Robustness score is positive when formula violated"""
        # Clear sustained violation
        entropy = [3.0, 3.5, 3.2, 3.1]

        result = self.formula.evaluate(entropy_trace=entropy)

        self.assertTrue(result.is_violation)
        self.assertGreater(result.robustness_score, 0.0)

    def test_robustness_score_negative_when_not_violated(self):
        """Test: Robustness score is negative when formula not violated"""
        entropy = [1.0, 1.5, 1.3]

        result = self.formula.evaluate(entropy_trace=entropy)

        self.assertFalse(result.is_violation)
        self.assertLess(result.robustness_score, 0.0)

    def test_violation_time_correct(self):
        """Test: Violation time points to start of sustained period"""
        entropy = [1.0, 1.5, 2.5, 2.8, 3.0, 1.0]
        #                     ^---- violation starts at t=2

        result = self.formula.evaluate(entropy_trace=entropy)

        self.assertTrue(result.is_violation)
        self.assertEqual(result.violation_time, 2)

    def test_empty_entropy_trace(self):
        """Test: Empty trace returns no violation"""
        result = self.formula.evaluate(entropy_trace=[])

        self.assertFalse(result.is_violation)


class TestDetachmentFormula(unittest.TestCase):
    """Test φ₂: Eventually Always(A(t) < θ_A)"""

    def setUp(self):
        self.params = STLFormulaParams(theta_A=0.4, T=100, w=3)
        self.formula = STLFormula('detachment', self.params)

    def test_sustained_low_attention_detected(self):
        """Test: Sustained low attention (3+ consecutive) triggers violation"""
        # Attention with sustained low period
        attention = [0.7, 0.6, 0.2, 0.3, 0.25, 0.7]

        result = self.formula.evaluate(attention_trace=attention)

        self.assertTrue(result.is_violation)
        self.assertEqual(result.formula_name, 'detachment')

    def test_no_sustained_low_attention(self):
        """Test: Brief dips don't trigger violation"""
        # Low attention but not sustained
        attention = [0.7, 0.3, 0.7, 0.2, 0.8]

        result = self.formula.evaluate(attention_trace=attention)

        self.assertFalse(result.is_violation)

    def test_all_high_attention(self):
        """Test: All high attention doesn't trigger violation"""
        attention = [0.7, 0.8, 0.75, 0.9]

        result = self.formula.evaluate(attention_trace=attention)

        self.assertFalse(result.is_violation)

    def test_violation_at_beginning(self):
        """Test: Violation can occur at start of trace"""
        attention = [0.2, 0.25, 0.3, 0.7, 0.8]

        result = self.formula.evaluate(attention_trace=attention)

        self.assertTrue(result.is_violation)
        self.assertEqual(result.violation_time, 0)

    def test_empty_attention_trace(self):
        """Test: Empty trace returns no violation"""
        result = self.formula.evaluate(attention_trace=[])

        self.assertFalse(result.is_violation)


class TestCombinedFormula(unittest.TestCase):
    """Test φ₃: Eventually Always(H(t) > θ_H AND A(t) < θ_A)"""

    def setUp(self):
        self.params = STLFormulaParams(theta_H=2.0, theta_A=0.4, T=100, w=3)
        self.formula = STLFormula('combined', self.params)

    def test_both_conditions_sustained_triggers_violation(self):
        """Test: Both high entropy AND low attention sustained triggers violation"""
        entropy =   [1.0, 1.5, 2.5, 2.8, 3.0, 1.5]
        attention = [0.7, 0.6, 0.2, 0.3, 0.25, 0.7]

        result = self.formula.evaluate(entropy_trace=entropy, attention_trace=attention)

        self.assertTrue(result.is_violation)
        self.assertEqual(result.formula_name, 'combined')

    def test_only_entropy_high_no_violation(self):
        """Test: Only high entropy (without low attention) doesn't trigger"""
        entropy =   [2.5, 2.8, 3.0, 2.6]
        attention = [0.7, 0.8, 0.75, 0.9]  # All high

        result = self.formula.evaluate(entropy_trace=entropy, attention_trace=attention)

        self.assertFalse(result.is_violation)

    def test_only_attention_low_no_violation(self):
        """Test: Only low attention (without high entropy) doesn't trigger"""
        entropy =   [1.0, 1.2, 1.1, 1.3]  # All low
        attention = [0.2, 0.3, 0.25, 0.28]

        result = self.formula.evaluate(entropy_trace=entropy, attention_trace=attention)

        self.assertFalse(result.is_violation)

    def test_conditions_not_simultaneous_no_violation(self):
        """Test: Conditions at different times don't trigger violation"""
        entropy =   [2.5, 2.8, 3.0, 1.0, 1.5, 1.2]
        attention = [0.7, 0.8, 0.9, 0.2, 0.3, 0.25]
        #            ^-- high entropy    ^-- low attention (but not simultaneous)

        result = self.formula.evaluate(entropy_trace=entropy, attention_trace=attention)

        self.assertFalse(result.is_violation)

    def test_perfect_attack_pattern(self):
        """Test: Clear attack pattern (high H, low A sustained) detected"""
        # Perfect attack: high entropy, low attention throughout
        entropy =   [2.8, 3.0, 2.9, 3.1, 2.7]
        attention = [0.2, 0.25, 0.3, 0.28, 0.22]

        result = self.formula.evaluate(entropy_trace=entropy, attention_trace=attention)

        self.assertTrue(result.is_violation)
        self.assertGreater(result.robustness_score, 0.0)

    def test_empty_traces(self):
        """Test: Empty traces return no violation"""
        result = self.formula.evaluate(entropy_trace=[], attention_trace=[])

        self.assertFalse(result.is_violation)


class TestRobustnessScores(unittest.TestCase):
    """Test quantitative robustness computation"""

    def test_higher_violation_gives_higher_robustness(self):
        """Test: Stronger violation gives higher robustness score"""
        params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        formula = STLFormula('waffling', params)

        # Weak violation (just above threshold)
        entropy_weak = [2.1, 2.2, 2.15]
        result_weak = formula.evaluate(entropy_trace=entropy_weak)

        # Strong violation (far above threshold)
        entropy_strong = [5.0, 5.5, 5.2]
        result_strong = formula.evaluate(entropy_trace=entropy_strong)

        self.assertTrue(result_weak.is_violation)
        self.assertTrue(result_strong.is_violation)
        self.assertGreater(result_strong.robustness_score, result_weak.robustness_score)

    def test_robustness_measures_minimum_in_window(self):
        """Test: Robustness is minimum excess in window (weakest link)"""
        params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        formula = STLFormula('waffling', params)

        # Window with varying excesses: 0.1, 1.0, 0.5
        entropy = [2.1, 3.0, 2.5]

        result = formula.evaluate(entropy_trace=entropy)

        # Robustness should be minimum: 2.1 - 2.0 = 0.1
        self.assertAlmostEqual(result.robustness_score, 0.1, delta=0.01)


class TestDatasetEvaluation(unittest.TestCase):
    """Test formula evaluation on full datasets"""

    def create_mock_dataset(self, n_attack=10, n_normal=10):
        """Helper to create mock traces"""
        traces = []
        validations = []

        # Attack traces (high entropy, low attention)
        for i in range(n_attack):
            traces.append({
                'entropy_trace': [2.8, 3.0, 2.9, 3.1] * 10,  # Sustained high
                'attention_trace': [0.2, 0.25, 0.3, 0.28] * 10  # Sustained low
            })
            validations.append({'is_hallucination': True})

        # Normal traces (low entropy, high attention)
        for i in range(n_normal):
            traces.append({
                'entropy_trace': [1.0, 1.2, 1.1, 1.3] * 10,
                'attention_trace': [0.7, 0.8, 0.75, 0.9] * 10
            })
            validations.append({'is_hallucination': False})

        return traces, validations

    def test_dataset_evaluation_structure(self):
        """Test: Dataset evaluation returns correct structure"""
        traces, validations = self.create_mock_dataset(5, 5)

        params = STLFormulaParams(theta_H=2.0, theta_A=0.4, T=100, w=3)
        formula = STLFormula('combined', params)

        results = evaluate_formula_on_dataset(formula, traces, validations)

        # Check structure
        self.assertIn('formula_type', results)
        self.assertIn('parameters', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('metrics', results)

        # Check confusion matrix
        self.assertIn('TP', results['confusion_matrix'])
        self.assertIn('FP', results['confusion_matrix'])
        self.assertIn('TN', results['confusion_matrix'])
        self.assertIn('FN', results['confusion_matrix'])

        # Check metrics
        self.assertIn('TPR', results['metrics'])
        self.assertIn('FPR', results['metrics'])
        self.assertIn('f1_score', results['metrics'])

    def test_perfect_classifier(self):
        """Test: Perfect separation gives perfect metrics"""
        traces, validations = self.create_mock_dataset(10, 10)

        params = STLFormulaParams(theta_H=2.0, theta_A=0.4, T=100, w=3)
        formula = STLFormula('combined', params)

        results = evaluate_formula_on_dataset(formula, traces, validations)

        # All attacks should be detected (TP=10)
        # All normal should be correct (TN=10)
        self.assertEqual(results['confusion_matrix']['TP'], 10)
        self.assertEqual(results['confusion_matrix']['TN'], 10)
        self.assertEqual(results['confusion_matrix']['FP'], 0)
        self.assertEqual(results['confusion_matrix']['FN'], 0)

        # Perfect metrics
        self.assertAlmostEqual(results['metrics']['TPR'], 1.0, delta=0.01)
        self.assertAlmostEqual(results['metrics']['FPR'], 0.0, delta=0.01)
        self.assertAlmostEqual(results['metrics']['f1_score'], 1.0, delta=0.01)

    def test_metrics_mathematical_consistency(self):
        """Test: Metrics are mathematically consistent"""
        traces, validations = self.create_mock_dataset(8, 12)

        params = STLFormulaParams(theta_H=2.0, theta_A=0.4, T=100, w=3)
        formula = STLFormula('combined', params)

        results = evaluate_formula_on_dataset(formula, traces, validations)

        cm = results['confusion_matrix']
        metrics = results['metrics']

        # TPR = TP / (TP + FN)
        expected_tpr = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0
        self.assertAlmostEqual(metrics['TPR'], expected_tpr, delta=0.01)

        # FPR = FP / (FP + TN)
        expected_fpr = cm['FP'] / (cm['FP'] + cm['TN']) if (cm['FP'] + cm['TN']) > 0 else 0.0
        self.assertAlmostEqual(metrics['FPR'], expected_fpr, delta=0.01)

        # Recall = TPR
        self.assertAlmostEqual(metrics['recall'], metrics['TPR'], delta=0.01)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_evaluate_without_required_trace(self):
        """Test: Evaluating without required trace raises error"""
        params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        formula = STLFormula('waffling', params)

        with self.assertRaises(ValueError):
            formula.evaluate(entropy_trace=None)

    def test_trace_shorter_than_window(self):
        """Test: Trace shorter than window size doesn't crash"""
        params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        formula = STLFormula('waffling', params)

        # Only 2 tokens (less than w=3)
        entropy = [3.0, 3.5]

        result = formula.evaluate(entropy_trace=entropy)

        # Should not violate (need w=3 consecutive)
        self.assertFalse(result.is_violation)

    def test_constant_signal(self):
        """Test: Constant signal works correctly"""
        params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        formula = STLFormula('waffling', params)

        # Constant high entropy
        entropy = [3.0] * 20

        result = formula.evaluate(entropy_trace=entropy)

        self.assertTrue(result.is_violation)

    def test_single_value_trace(self):
        """Test: Single value trace doesn't crash"""
        params = STLFormulaParams(theta_H=2.0, T=100, w=3)
        formula = STLFormula('waffling', params)

        entropy = [3.0]

        result = formula.evaluate(entropy_trace=entropy)

        self.assertFalse(result.is_violation)  # Need w=3


if __name__ == '__main__':
    unittest.main()
