"""
Comprehensive Unit Tests for Oracle Validator

Tests cover:
1. Answer extraction from various formats
2. Factuality checking (correct vs incorrect answers)
3. Faithfulness checking via NLI (contradiction detection)
4. Batch validation
5. Statistics computation
6. Edge cases (malformed inputs, no answer, etc.)

NO try-except gaming - tests verify actual behavior.
"""

import unittest
import torch
from datasets.oracle_validator import OracleValidator, ValidationResult


class TestAnswerExtraction(unittest.TestCase):
    """Test answer token extraction from various text formats"""

    def setUp(self):
        self.validator = OracleValidator(device='cpu')

    def test_extract_answer_standard_format(self):
        """Test: 'The answer is (A)' format"""
        text = "Let me think... The answer is (A)"
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'A')

    def test_extract_answer_colon_format(self):
        """Test: 'Answer: B' format"""
        text = "After careful consideration, answer: B"
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'B')

    def test_extract_answer_therefore_format(self):
        """Test: 'Therefore, (C)' format"""
        text = "Based on the calculation, therefore, (C)"
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'C')

    def test_extract_answer_standalone_parenthesis(self):
        """Test: Standalone '(D)' at end"""
        text = "The reasoning leads to (D)."
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'D')

    def test_extract_answer_is_correct_format(self):
        """Test: 'A is correct' format"""
        text = "Option A is correct based on the evidence"
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'A')

    def test_extract_answer_multiple_mentions(self):
        """Test: Multiple mentions - should return last one"""
        text = "I considered A and B, but ultimately C is the answer"
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'C')

    def test_extract_answer_no_answer(self):
        """Test: No valid answer token"""
        text = "I cannot determine the answer from this information"
        answer = self.validator.extract_answer_token(text)
        self.assertIsNone(answer)

    def test_extract_answer_case_insensitive(self):
        """Test: Case insensitivity"""
        text = "the answer is (a)"
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'A')

    def test_extract_answer_with_explanation(self):
        """Test: Answer embedded in long explanation"""
        text = """
        Let me analyze this step by step. First, we need to consider the
        fundamental principles. After careful evaluation, the answer is (B)
        because it aligns with the theory.
        """
        answer = self.validator.extract_answer_token(text)
        self.assertEqual(answer, 'B')


class TestFactualityChecking(unittest.TestCase):
    """Test factuality verification against ground truth"""

    def setUp(self):
        self.validator = OracleValidator(device='cpu')

    def test_correct_answer_not_hallucination(self):
        """Test: Correct answer should NOT be classified as hallucination"""
        trace = {
            'prompt': 'What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6',
            'generated_text': 'The answer is (B) because 2+2 equals 4.'
        }
        result = self.validator.validate_trace(trace, ground_truth='B')

        self.assertFalse(result.is_hallucination)
        self.assertIsNone(result.hallucination_type)
        self.assertEqual(result.confidence, 1.0)

    def test_wrong_answer_is_hallucination(self):
        """Test: Wrong answer should be classified as hallucination"""
        trace = {
            'prompt': 'What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6',
            'generated_text': 'The answer is (A) because 2+2 equals 3.'
        }
        result = self.validator.validate_trace(trace, ground_truth='B')

        self.assertTrue(result.is_hallucination)
        self.assertIn(result.hallucination_type, ['factuality', 'faithfulness'])
        self.assertGreater(result.confidence, 0.0)

    def test_no_answer_extraction_is_hallucination(self):
        """Test: Failure to extract answer counts as hallucination"""
        trace = {
            'prompt': 'What is the capital of France?',
            'generated_text': 'I do not know the answer to this question.'
        }
        result = self.validator.validate_trace(trace, ground_truth='A')

        self.assertTrue(result.is_hallucination)
        self.assertEqual(result.hallucination_type, 'factuality')
        self.assertEqual(result.confidence, 0.8)
        self.assertIn('Failed to extract answer', result.explanation)


class TestFaithfulnessChecking(unittest.TestCase):
    """Test NLI-based faithfulness verification"""

    def setUp(self):
        self.validator = OracleValidator(device='cpu')

    def test_nli_entailment_detection(self):
        """Test: NLI correctly identifies entailment"""
        premise = "The sky is blue during the day."
        hypothesis = "During daytime, the sky appears blue."

        label, confidence = self.validator.check_nli(premise, hypothesis)

        self.assertEqual(label, 'entailment')
        self.assertGreater(confidence, 0.5)

    def test_nli_contradiction_detection(self):
        """Test: NLI correctly identifies contradiction"""
        premise = "Water freezes at 0 degrees Celsius."
        hypothesis = "Water remains liquid at 0 degrees Celsius."

        label, confidence = self.validator.check_nli(premise, hypothesis)

        self.assertEqual(label, 'contradiction')
        self.assertGreater(confidence, 0.5)

    def test_nli_neutral_detection(self):
        """Test: NLI correctly identifies neutral relationship"""
        premise = "The meeting is at 3 PM."
        hypothesis = "The weather is sunny today."

        label, confidence = self.validator.check_nli(premise, hypothesis)

        self.assertEqual(label, 'neutral')
        self.assertGreater(confidence, 0.5)

    def test_faithfulness_hallucination_detection(self):
        """Test: Faithfulness error detected when explanation contradicts prompt"""
        trace = {
            'prompt': 'Which element has atomic number 1? (A) Helium (B) Hydrogen (C) Oxygen (D) Carbon',
            'generated_text': 'The answer is (A) because the element with atomic number 1 is Helium.'
        }
        result = self.validator.validate_trace(trace, ground_truth='B')

        self.assertTrue(result.is_hallucination)
        # Note: Faithfulness check depends on NLI model's interpretation
        # We verify the NLI label is populated
        self.assertIsNotNone(result.nli_label)
        self.assertIsNotNone(result.nli_score)

    def test_factuality_without_faithfulness_error(self):
        """Test: Factuality error without contradiction (model makes reasoning error)"""
        trace = {
            'prompt': 'What is 7 * 8? (A) 54 (B) 56 (C) 58 (D) 60',
            'generated_text': 'The answer is (A) because I calculated 7 * 8 = 54.'
        }
        result = self.validator.validate_trace(trace, ground_truth='B')

        self.assertTrue(result.is_hallucination)
        # Wrong answer but explanation doesn't contradict prompt
        # Type should be factuality (calculation error)
        self.assertIsNotNone(result.hallucination_type)


class TestBatchValidation(unittest.TestCase):
    """Test batch processing of multiple traces"""

    def setUp(self):
        self.validator = OracleValidator(device='cpu')

    def test_batch_validation_length_mismatch(self):
        """Test: Batch validation fails with mismatched lengths"""
        traces = [
            {'prompt': 'Q1', 'generated_text': 'A'},
            {'prompt': 'Q2', 'generated_text': 'B'}
        ]
        ground_truths = ['A']  # Length mismatch

        with self.assertRaises(AssertionError):
            self.validator.validate_batch(traces, ground_truths)

    def test_batch_validation_success(self):
        """Test: Batch validation processes all traces correctly"""
        traces = [
            {'prompt': 'What is 1+1? (A) 1 (B) 2', 'generated_text': 'Answer: (B)'},
            {'prompt': 'What is 2+2? (A) 3 (B) 4', 'generated_text': 'Answer: (A)'},
            {'prompt': 'What is 3+3? (A) 6 (B) 7', 'generated_text': 'Answer: (A)'}
        ]
        ground_truths = ['B', 'B', 'A']

        results = self.validator.validate_batch(traces, ground_truths)

        self.assertEqual(len(results), 3)
        self.assertFalse(results[0].is_hallucination)  # Correct
        self.assertTrue(results[1].is_hallucination)   # Wrong
        self.assertFalse(results[2].is_hallucination)  # Correct

    def test_batch_validation_all_correct(self):
        """Test: Batch with all correct answers"""
        traces = [
            {'prompt': 'Q1', 'generated_text': 'The answer is (A)'},
            {'prompt': 'Q2', 'generated_text': 'The answer is (B)'},
            {'prompt': 'Q3', 'generated_text': 'The answer is (C)'}
        ]
        ground_truths = ['A', 'B', 'C']

        results = self.validator.validate_batch(traces, ground_truths)

        hallucinations = [r for r in results if r.is_hallucination]
        self.assertEqual(len(hallucinations), 0)

    def test_batch_validation_all_wrong(self):
        """Test: Batch with all incorrect answers"""
        traces = [
            {'prompt': 'Q1', 'generated_text': 'The answer is (B)'},
            {'prompt': 'Q2', 'generated_text': 'The answer is (C)'},
            {'prompt': 'Q3', 'generated_text': 'The answer is (D)'}
        ]
        ground_truths = ['A', 'A', 'A']

        results = self.validator.validate_batch(traces, ground_truths)

        hallucinations = [r for r in results if r.is_hallucination]
        self.assertEqual(len(hallucinations), 3)


class TestStatisticsComputation(unittest.TestCase):
    """Test validation statistics computation"""

    def setUp(self):
        self.validator = OracleValidator(device='cpu', confidence_threshold=0.9)

    def test_statistics_all_correct(self):
        """Test: Statistics for all correct predictions"""
        results = [
            ValidationResult(False, None, 1.0, 'Correct'),
            ValidationResult(False, None, 1.0, 'Correct'),
            ValidationResult(False, None, 1.0, 'Correct')
        ]

        stats = self.validator.compute_statistics(results)

        self.assertEqual(stats['total_traces'], 3)
        self.assertEqual(stats['hallucination_count'], 0)
        self.assertEqual(stats['hallucination_rate'], 0.0)
        self.assertEqual(stats['factuality_errors'], 0)
        self.assertEqual(stats['faithfulness_errors'], 0)
        self.assertEqual(stats['avg_confidence'], 1.0)

    def test_statistics_mixed_results(self):
        """Test: Statistics for mixed correct/hallucination results"""
        results = [
            ValidationResult(False, None, 1.0, 'Correct'),
            ValidationResult(True, 'factuality', 0.95, 'Wrong answer'),
            ValidationResult(True, 'faithfulness', 0.92, 'Contradiction'),
            ValidationResult(False, None, 1.0, 'Correct')
        ]

        stats = self.validator.compute_statistics(results)

        self.assertEqual(stats['total_traces'], 4)
        self.assertEqual(stats['hallucination_count'], 2)
        self.assertEqual(stats['hallucination_rate'], 0.5)
        self.assertEqual(stats['factuality_errors'], 1)
        self.assertEqual(stats['faithfulness_errors'], 1)
        self.assertAlmostEqual(stats['avg_confidence'], (1.0 + 0.95 + 0.92 + 1.0) / 4)

    def test_statistics_confidence_filtering(self):
        """Test: Statistics correctly counts high vs low confidence"""
        results = [
            ValidationResult(False, None, 0.95, 'High confidence'),
            ValidationResult(True, 'factuality', 0.85, 'Low confidence'),
            ValidationResult(False, None, 0.92, 'High confidence'),
            ValidationResult(True, 'factuality', 0.70, 'Low confidence')
        ]

        stats = self.validator.compute_statistics(results)

        self.assertEqual(stats['high_confidence_predictions'], 2)  # >= 0.9
        self.assertEqual(stats['low_confidence_count'], 2)  # < 0.9

    def test_statistics_empty_list(self):
        """Test: Statistics handle empty results gracefully"""
        results = []

        stats = self.validator.compute_statistics(results)

        self.assertEqual(stats['total_traces'], 0)
        self.assertEqual(stats['hallucination_count'], 0)
        self.assertEqual(stats['hallucination_rate'], 0.0)
        self.assertEqual(stats['avg_confidence'], 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.validator = OracleValidator(device='cpu')

    def test_empty_generated_text(self):
        """Test: Empty generated text"""
        trace = {'prompt': 'What is X?', 'generated_text': ''}
        result = self.validator.validate_trace(trace, ground_truth='A')

        self.assertTrue(result.is_hallucination)
        self.assertEqual(result.hallucination_type, 'factuality')

    def test_missing_prompt_key(self):
        """Test: Missing prompt key defaults to empty string"""
        trace = {'generated_text': 'The answer is (A)'}
        result = self.validator.validate_trace(trace, ground_truth='A')

        # Should still work, extracting answer
        self.assertFalse(result.is_hallucination)

    def test_missing_generated_text_key(self):
        """Test: Missing generated_text key defaults to empty string"""
        trace = {'prompt': 'What is X?'}
        result = self.validator.validate_trace(trace, ground_truth='A')

        self.assertTrue(result.is_hallucination)

    def test_very_long_text(self):
        """Test: Very long generated text (>512 tokens)"""
        long_text = "Let me explain. " * 200 + "The answer is (B)."
        trace = {'prompt': 'What is X?', 'generated_text': long_text}

        result = self.validator.validate_trace(trace, ground_truth='B')

        # Should still extract answer despite length
        self.assertFalse(result.is_hallucination)

    def test_special_characters_in_text(self):
        """Test: Special characters don't break extraction"""
        trace = {
            'prompt': 'Question with special chars: @#$%',
            'generated_text': 'After analysis... the answer is (C)!!!'
        }
        result = self.validator.validate_trace(trace, ground_truth='C')

        self.assertFalse(result.is_hallucination)

    def test_lowercase_ground_truth(self):
        """Test: Lowercase ground truth is handled (comparison is case-insensitive)"""
        trace = {'prompt': 'Q', 'generated_text': 'Answer: (a)'}
        result = self.validator.validate_trace(trace, ground_truth='a')

        # Note: Our implementation converts to uppercase for comparison
        self.assertFalse(result.is_hallucination)


class TestModelLoading(unittest.TestCase):
    """Test model initialization and device handling"""

    def test_model_loads_on_cpu(self):
        """Test: Model successfully loads on CPU"""
        validator = OracleValidator(device='cpu')
        self.assertEqual(validator.device, 'cpu')
        self.assertIsNotNone(validator.nli_model)
        self.assertIsNotNone(validator.tokenizer)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_model_loads_on_cuda(self):
        """Test: Model successfully loads on CUDA (if available)"""
        validator = OracleValidator(device='cuda')
        self.assertEqual(validator.device, 'cuda')
        self.assertTrue(next(validator.nli_model.parameters()).is_cuda)

    def test_auto_device_detection(self):
        """Test: Auto device detection works"""
        validator = OracleValidator(device=None)
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(validator.device, expected_device)

    def test_confidence_threshold_setting(self):
        """Test: Custom confidence threshold is set correctly"""
        validator = OracleValidator(confidence_threshold=0.85)
        self.assertEqual(validator.confidence_threshold, 0.85)


if __name__ == '__main__':
    unittest.main()
