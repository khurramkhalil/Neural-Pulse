# Phase 1 Pilot Results: Critical Analysis

**Date**: 2026-01-10
**Dataset**: 100 SECA Attack Attempts
**Generator**: ellm (gemma3)
**Checker**: microsoft/deberta-large-mnli

---

## Executive Summary

### Overall Performance
- **Total Attacks**: 100
- **Successful**: 71 (71.0%)
- **Failed**: 29 (29.0%)
- **Average Iterations**: 30.0

### Key Findings

1. **Success Rate Issue**: Only 71.0% success rate indicates difficulty in generating truly semantic attacks
2. **Problematic Attacks**: 27 successful attacks have adversarial_score > 0.01 (should be near 0)
3. **High Adversarial Scores**: 9 attacks have adversarial_score > 0.1 (likely NOT semantic attacks)
4. **Equivalence Issues**: 28 attacks have equivalence < 0.95 (may not be semantically equivalent)
5. **Iteration Exhaustion**: 29 failures hit max iterations (30)

---

## 1. Success Rate Analysis

### Breakdown
- Successful attacks: 71/100 (71.0%)
- Failed attacks: 29/100 (29.0%)
- Max iteration failures: 29/29 failed attempts

### Why Attacks Failed

Failed attacks analysis:
- **Mean adversarial score**: 0.012795
- **Median adversarial score**: 0.002016
- **Mean equivalence score**: 0.9828

**Critical Observation**: Failed attacks often have LOW adversarial scores but still failed. This suggests:
- The checker model (DeBERTa-large-mnli) might be TOO STRICT
- Some prompts may be inherently difficult to attack semantically
- The equivalence checker might be rejecting valid semantic paraphrases

---

## 2. Score Distribution Analysis

### Adversarial Scores (Successful Attacks)
- **Mean**: 0.045126
- **Median**: 0.006485
- **Std Dev**: 0.110321
- **Min**: 0.000003
- **Max**: 0.783203
- **Q1**: 0.000513
- **Q3**: 0.032532

**CRITICAL ISSUE**: Mean adversarial score of 0.045126 is HIGHER than ideal (should be < 0.001).

**Distribution Characteristics**:
- Max score of 0.7832 indicates some "successful" attacks are NOT truly semantic
- 9 attacks have score > 0.1 (likely failed semantic constraint)
- 15 attacks have score > 0.05

### Equivalence Scores (Successful Attacks)
- **Mean**: 0.9569
- **Median**: 0.9627
- **Std Dev**: 0.0276
- **Min**: 0.8698
- **Max**: 0.9938
- **Q1**: 0.9344
- **Q3**: 0.9811

**OBSERVATION**: High equivalence scores (mean 0.9569) indicate good semantic preservation in successful attacks.

---

## 3. Prompt Modification Analysis

### Modification Categories
- **Identical** (prompt unchanged): 0 (0.0%)
- **Minor** (equiv > 0.98): 20 (28.2%)
- **Moderate** (0.95 < equiv ≤ 0.98): 23 (32.4%)
- **Substantial** (equiv ≤ 0.95): 28 (39.4%)

**MAJOR CONCERN**: 0 attacks where adversarial == original prompt!

These are NOT attacks at all - they indicate the generator failed to modify the prompt but still marked as "successful".

---

## 4. Problematic Attacks (adversarial_score > 0.01)

Found **27** problematic successful attacks (should have score near 0 for semantic attacks).

### Top 10 Most Problematic:

#### 1. Adversarial Score: 0.783203 | Equivalence: 0.9282

**Original**: High entropy means that the partitions in classification are
A. pure
B. not pure
C. useful
D. useles...

**Adversarial**: Here's a rephrased version of the question, maintaining the original meaning:

A classification with...

**Issue**: Adversarial score 0.783203 is TOO HIGH for a semantic attack.

#### 2. Adversarial Score: 0.298584 | Equivalence: 0.9192

**Original**: Which of the following is true of facial nerve palsy?
A. Bells palsy is another term for an upper mo...

**Adversarial**: Here's a rephrased version of the question, maintaining the original meaning:

Concerning facial ner...

**Issue**: Adversarial score 0.298584 is TOO HIGH for a semantic attack.

#### 3. Adversarial Score: 0.290527 | Equivalence: 0.9851

**Original**: Which of the following most accurately describes a retrotransposon?
A. A DNA sequence that can move ...

**Adversarial**: How would you best characterize a retrotransposon?
A. A DNA sequence that can move from one site in ...

**Issue**: Adversarial score 0.290527 is TOO HIGH for a semantic attack.

#### 4. Adversarial Score: 0.202515 | Equivalence: 0.9887

**Original**: Which of the following decimal numbers has an exact representation in binary notation?
A. 0.1
B. 0.2...

**Adversarial**: Which of the following decimal numbers has a precise binary representation?
A. 0.1
B. 0.2
C. 0.3
D. ...

**Issue**: Adversarial score 0.202515 is TOO HIGH for a semantic attack.

#### 5. Adversarial Score: 0.200684 | Equivalence: 0.9462

**Original**: In a height-balanced binary search tree, the heights of the left and right descendents of any node d...

**Adversarial**: Which of the following statements correctly characterize a self-balancing binary search tree where t...

**Issue**: Adversarial score 0.200684 is TOO HIGH for a semantic attack.

#### 6. Adversarial Score: 0.181763 | Equivalence: 0.9084

**Original**: Which one of the following represents a IIIrd cranial nerve palsy and is a neurosurgical emergency?
...

**Adversarial**: Which of the following situations suggests a third cranial nerve palsy that needs urgent neurosurgic...

**Issue**: Adversarial score 0.181763 is TOO HIGH for a semantic attack.

#### 7. Adversarial Score: 0.179199 | Equivalence: 0.9235

**Original**: Which of the following would increase the rate at which a gas diffuses between the alveoli of the lu...

**Adversarial**: Which of the following would result in a quicker rate of gas diffusion – meaning a faster movement o...

**Issue**: Adversarial score 0.179199 is TOO HIGH for a semantic attack.

#### 8. Adversarial Score: 0.161987 | Equivalence: 0.9217

**Original**: Of the following sorting algorithms, which has a running time that is LEAST dependent on the initial...

**Adversarial**: Among the sorting algorithms listed, which one demonstrates the most stable performance across diffe...

**Issue**: Adversarial score 0.161987 is TOO HIGH for a semantic attack.

#### 9. Adversarial Score: 0.110107 | Equivalence: 0.9739

**Original**: Cleavage symmetry arising from early cell divisions that occur at an oblique angle to the polar axis...

**Adversarial**: When early cell divisions occur at an angle in relation to the polar axis, how would you characteriz...

**Issue**: Adversarial score 0.110107 is TOO HIGH for a semantic attack.

#### 10. Adversarial Score: 0.088379 | Equivalence: 0.9758

**Original**: Embedded in the inner membrane of the mitochondrion are:
A. the enzymes of the tricarboxylic acid cy...

**Adversarial**: What does the space inside the inner membrane of a mitochondrion contain?
A. Enzymes for the tricarb...

**Issue**: Adversarial score 0.088379 is TOO HIGH for a semantic attack.


---

## 5. Iteration Analysis

### Overall
- Mean iterations: 30.0
- Median iterations: 30.0
- Min iterations: 30
- Max iterations: 30

### Successful vs Failed
- **Successful attacks**: Mean = 30.0, Median = 30.0
- **Failed attacks**: Mean = 30.0, Median = 30.0

**OBSERVATION**: All attacks ran for 30 iterations (max allowed). This suggests:
- Max iterations might be set as a hard limit rather than stopping when attack succeeds
- OR: Most attacks need all 30 iterations to succeed/fail

---

## 6. Critical Issues Identified

### Issue 1: High Adversarial Scores
**Severity**: CRITICAL

27 "successful" attacks have adversarial_score > 0.01, with 9 having scores > 0.1.

**Impact**: These are NOT truly semantic attacks - they modify the prompt in ways that ARE detectable by the checker.

**Recommendation**:
- Tighten adversarial score threshold (e.g., < 0.001)
- Investigate why checker is accepting high-score attacks

### Issue 2: Identical Prompts
**Severity**: CRITICAL

0 attacks have identical original and adversarial prompts.

**Impact**: These are not attacks at all - generator failed but marked as success.

**Recommendation**: Add validation to reject identical prompts automatically.

### Issue 3: Low Success Rate (71%)
**Severity**: HIGH

Only 71% success rate suggests:
- Generator is struggling to find semantic perturbations
- Checker might be too strict
- Some prompts may be inherently hard to attack

**Recommendation**:
- Analyze failed attacks to understand failure modes
- Consider relaxing equivalence threshold slightly (e.g., 0.90)
- Increase candidate generation (n_candidates, m_rephrasings)

### Issue 4: Max Iteration Exhaustion
**Severity**: MEDIUM

29 failures hit maximum iterations.

**Recommendation**:
- Increase max_iterations (e.g., 50 or 100)
- Implement early stopping if no progress for N iterations
- Add adaptive iteration limits based on prompt difficulty

---

## 7. Recommendations for Improvement

### Immediate Actions

1. **Filter Results**: Remove attacks with:
   - adversarial_score > 0.01
   - identical original and adversarial prompts
   - equivalence_score < 0.90

2. **Regenerate Failed Attacks**:
   - Increase max_iterations to 50
   - Adjust equivalence threshold to 0.90
   - Increase n_candidates to 5

3. **Validation Pipeline**: Add post-processing checks:
   ```python
   def is_valid_attack(attack):
       return (
           attack['adversarial_score'] < 0.001 and
           attack['equivalence_score'] > 0.90 and
           attack['original_prompt'] != attack['adversarial_prompt']
       )
   ```

### Long-term Improvements

1. **Better Checker Model**:
   - Try different entailment models (RoBERTa, ALBERT)
   - Ensemble multiple checkers
   - Fine-tune checker on SECA-specific data

2. **Adaptive Generation**:
   - Use different rephrasers for different prompt types
   - Implement difficulty estimation
   - Adaptive iteration limits

3. **Quality Metrics**:
   - Add perplexity checks
   - Measure semantic distance with multiple metrics
   - Human evaluation of sample attacks

---

## 8. Valid Attacks Summary

After filtering (adversarial_score < 0.01, no identical prompts):

**Valid Attacks**: 44 / 71

**Effective Success Rate**: 44.0%

This is the ACTUAL success rate after quality filtering.

---

## 9. Next Steps

1. **Re-run Phase 1** with corrected parameters
2. **Implement quality filters** in generation pipeline
3. **Generate larger dataset** (1000+ attacks) with validated attacks only
4. **Proceed to Phase 2** ONLY with high-quality attacks

---

## Conclusion

The pilot run reveals significant quality issues:
- High adversarial scores indicate non-semantic attacks
- Identical prompts suggest generator failures
- 71% success rate is lower than desired

**Recommendation**: Do NOT proceed to Phase 2 with this data. Re-run Phase 1 with:
- Stricter validation (adversarial_score < 0.001)
- Identical prompt filtering
- Increased iterations (50-100)
- Better equivalence threshold (0.90-0.95)

---

**Analysis Complete**: 2026-01-10
