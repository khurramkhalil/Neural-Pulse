"""
Oracle Validator for Neural Pulse

Labels traces by ACTUAL outcome (hallucination vs correct) rather than prompt type.
Uses DeBERTa-v3-large NLI model for faithfulness checking and ground truth for factuality.

Key Design Decisions:
1. Oracle-only approach (hybrid validation moved to future work)
2. Two-step validation: Factuality check → Faithfulness check (NLI)
3. Confidence threshold: 0.9 (high-confidence predictions only)
4. Hallucination taxonomy: factuality vs faithfulness errors
"""

import torch
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured validation output"""
    is_hallucination: bool
    hallucination_type: Optional[str]  # 'factuality' | 'faithfulness' | None
    confidence: float
    explanation: str
    nli_label: Optional[str] = None  # 'entailment' | 'neutral' | 'contradiction'
    nli_score: Optional[float] = None


class OracleValidator:
    """
    Oracle-based validator using DeBERTa-v3-large NLI model.

    Validation Pipeline:
    1. Extract answer token from generated text
    2. Check factuality: Does answer match ground truth?
    3. If incorrect, check faithfulness: Does explanation contradict prompt?
    4. Return structured validation result with confidence

    Example:
        validator = OracleValidator()
        result = validator.validate_trace(
            trace={'prompt': 'What is 2+2?', 'generated_text': 'The answer is 5...'},
            ground_truth='4'
        )
        print(f"Hallucination: {result.is_hallucination}, Type: {result.hallucination_type}")
    """

    def __init__(
        self,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        confidence_threshold: float = 0.9,
        device: Optional[str] = None
    ):
        """
        Initialize Oracle Validator.

        Args:
            nli_model_name: HuggingFace model for NLI (default: DeBERTa-v3-large)
            confidence_threshold: Minimum confidence for predictions (default: 0.9)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading NLI model: {nli_model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.to(self.device)
        self.nli_model.eval()

        # DeBERTa MNLI label mapping
        self.label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

        logger.info(f"Oracle Validator initialized (threshold={confidence_threshold})")

    def extract_answer_token(self, generated_text: str) -> Optional[str]:
        """
        Extract final answer from generated text.

        Supports multiple formats:
        - "The answer is X"
        - "Therefore, X"
        - "X is the correct answer"
        - Multiple choice: (A), (B), (C), (D)

        Args:
            generated_text: Full generated response

        Returns:
            Extracted answer token or None if not found
        """
        text = generated_text.strip()

        # Pattern 1: "The answer is X" or "answer: X"
        match = re.search(r'(?:the\s+)?answer\s+(?:is|:)\s+\(?([A-D])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 2: "Therefore, (X)" or "Thus, X"
        match = re.search(r'(?:therefore|thus|hence),?\s+\(?([A-D])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 3: Standalone "(X)" at end
        match = re.search(r'\(([A-D])\)\s*\.?\s*$', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 4: "X is correct" or "X is the answer"
        match = re.search(r'\(?([A-D])\)?\s+is\s+(?:the\s+)?(?:correct|answer)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 5: Last mentioned option letter
        matches = re.findall(r'\(?([A-D])\)?', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

        return None

    def check_nli(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Check entailment relationship using DeBERTa NLI.

        Args:
            premise: Context/prompt (what should be true)
            hypothesis: Generated explanation (what model claims)

        Returns:
            Tuple of (label, confidence_score)
            - label: 'entailment' | 'neutral' | 'contradiction'
            - confidence_score: Probability of predicted label [0, 1]
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_class].item()

        label = self.label_map[predicted_class]
        return label, confidence

    def validate_trace(
        self,
        trace: Dict,
        ground_truth: str
    ) -> ValidationResult:
        """
        Validate a single trace to determine if it's a hallucination.

        Validation Logic:
        1. Extract answer from generated_text
        2. If answer == ground_truth → NOT hallucination (confidence=1.0)
        3. If answer != ground_truth:
           a. Check faithfulness via NLI (prompt vs explanation)
           b. If contradiction → FAITHFULNESS hallucination
           c. Otherwise → FACTUALITY hallucination

        Args:
            trace: Dict with keys 'prompt', 'generated_text'
            ground_truth: Correct answer (e.g., 'A', 'B', 'C', 'D' for MMLU)

        Returns:
            ValidationResult with hallucination classification and confidence
        """
        prompt = trace.get('prompt', '')
        generated_text = trace.get('generated_text', '')

        # Step 1: Extract answer
        extracted_answer = self.extract_answer_token(generated_text)

        if extracted_answer is None:
            # Cannot extract answer - likely hallucination (no coherent response)
            return ValidationResult(
                is_hallucination=True,
                hallucination_type='factuality',
                confidence=0.8,  # Medium confidence (heuristic)
                explanation="Failed to extract answer token from response"
            )

        # Step 2: Factuality check
        is_factually_correct = (extracted_answer == ground_truth.upper())

        if is_factually_correct:
            # Correct answer → NOT hallucination
            return ValidationResult(
                is_hallucination=False,
                hallucination_type=None,
                confidence=1.0,
                explanation=f"Answer '{extracted_answer}' matches ground truth '{ground_truth}'"
            )

        # Step 3: Faithfulness check (answer is wrong, but why?)
        # Check if explanation contradicts the prompt
        nli_label, nli_confidence = self.check_nli(
            premise=prompt,
            hypothesis=generated_text
        )

        # Determine hallucination type
        if nli_label == 'contradiction':
            hallucination_type = 'faithfulness'
            explanation = f"Answer '{extracted_answer}' is wrong AND explanation contradicts prompt (NLI: {nli_label})"
        else:
            hallucination_type = 'factuality'
            explanation = f"Answer '{extracted_answer}' is wrong (expected '{ground_truth}'), but explanation is consistent with prompt (NLI: {nli_label})"

        return ValidationResult(
            is_hallucination=True,
            hallucination_type=hallucination_type,
            confidence=nli_confidence,
            explanation=explanation,
            nli_label=nli_label,
            nli_score=nli_confidence
        )

    def validate_batch(
        self,
        traces: List[Dict],
        ground_truths: List[str]
    ) -> List[ValidationResult]:
        """
        Validate multiple traces.

        Args:
            traces: List of trace dicts
            ground_truths: List of correct answers (aligned with traces)

        Returns:
            List of ValidationResult objects
        """
        assert len(traces) == len(ground_truths), "Traces and ground truths must have same length"

        results = []
        for i, (trace, gt) in enumerate(zip(traces, ground_truths)):
            logger.info(f"Validating trace {i+1}/{len(traces)}")
            result = self.validate_trace(trace, gt)
            results.append(result)

        return results

    def compute_statistics(self, results: List[ValidationResult]) -> Dict:
        """
        Compute validation statistics.

        Args:
            results: List of ValidationResult objects

        Returns:
            Dict with hallucination rates, confidence distribution, etc.
        """
        total = len(results)
        hallucinations = [r for r in results if r.is_hallucination]
        n_hallucinations = len(hallucinations)

        factuality_errors = [r for r in hallucinations if r.hallucination_type == 'factuality']
        faithfulness_errors = [r for r in hallucinations if r.hallucination_type == 'faithfulness']

        high_confidence = [r for r in results if r.confidence >= self.confidence_threshold]

        return {
            'total_traces': total,
            'hallucination_count': n_hallucinations,
            'hallucination_rate': n_hallucinations / total if total > 0 else 0.0,
            'factuality_errors': len(factuality_errors),
            'faithfulness_errors': len(faithfulness_errors),
            'high_confidence_predictions': len(high_confidence),
            'avg_confidence': sum(r.confidence for r in results) / total if total > 0 else 0.0,
            'low_confidence_count': total - len(high_confidence)
        }


def main():
    """CLI interface for oracle validation"""
    import argparse

    parser = argparse.ArgumentParser(description='Oracle Validator for Neural Pulse')
    parser.add_argument('--traces', type=str, required=True, help='Path to traces JSON file')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load data
    with open(args.traces, 'r') as f:
        traces = json.load(f)

    with open(args.ground_truth, 'r') as f:
        ground_truth_data = json.load(f)

    # Initialize validator
    validator = OracleValidator(
        confidence_threshold=args.confidence_threshold,
        device=args.device
    )

    # Validate
    ground_truths = [gt['answer'] for gt in ground_truth_data]
    results = validator.validate_batch(traces, ground_truths)

    # Compute statistics
    stats = validator.compute_statistics(results)

    # Save results
    output = {
        'validation_results': [
            {
                'trace_id': i,
                'is_hallucination': r.is_hallucination,
                'hallucination_type': r.hallucination_type,
                'confidence': r.confidence,
                'explanation': r.explanation,
                'nli_label': r.nli_label,
                'nli_score': r.nli_score
            }
            for i, r in enumerate(results)
        ],
        'statistics': stats
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Validation complete. Results saved to {args.output}")
    logger.info(f"Statistics: {stats}")


if __name__ == '__main__':
    main()
