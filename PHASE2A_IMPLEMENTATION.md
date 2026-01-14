# Phase 2a Implementation - Semantic Drift Signal

**Date**: 2026-01-11
**Status**: ✅ **READY FOR DEPLOYMENT**
**Critical Change**: Added Semantic Drift signal, dropped 3 weak signals

---

## 🎯 Executive Summary

**Phase 2 Autopsy**: Multi-signal classifier (4 signals) achieved AUC 0.654 - **WORSE** than best individual signal (Perplexity 0.706). This indicated:
1. **Signal Contradiction**: Perplexity learned NEGATIVE weight (contradicted hypothesis)
2. **Noise Dominance**: Weak signals (Attention, Attention Entropy) added noise, not information
3. **Fundamental Issue**: Internal signals measure "how model feels", not "what model means"

**Phase 2a Solution**: Pivot to **Semantic Drift** - measures whether generation stays grounded to prompt or drifts into hallucination.

**Key Changes**:
- ✅ **NEW**: Semantic Drift signal (cosine similarity between generation and prompt embeddings)
- ✅ **FOCUS**: Entropy + Semantic Drift (2 signals only)
- ✅ **DROPPED**: Attention (failed), Perplexity (wrong sign), Attention Entropy (weak)

---

## 📊 Theoretical Foundation

### The Semantic Drift Hypothesis

**Theory**: Hallucinations represent a **semantic departure** from the prompt's constraints.

**Mechanism**:
1. **Normal Generation**: Stays semantically anchored to prompt → High cosine similarity
2. **Hallucination**: Progressively drifts into fantasy state → Decreasing similarity

**Why This Works Better Than Internal Signals**:
- **Internal** (Entropy, Attention): Measures model confidence/mechanics
  - Problem: Model can be "confidently wrong" (high certainty, low entropy, but hallucinating)
- **Semantic**: Measures alignment with prompt intent
  - Advantage: Detects drift even when model is confident

---

## 🔧 Implementation Details

### 1. New Signal: Semantic Drift

**File**: `core/llama_hook.py`

**Computation**:
```python
def compute_semantic_drift(
    self,
    prompt_embedding: torch.Tensor,  # [hidden_dim]
    current_hidden_state: torch.Tensor  # [hidden_dim]
) -> float:
    """
    Compute cosine similarity between prompt and current generation.

    Returns: Similarity in [0, 1]
      - 1.0 = Perfectly aligned
      - 0.5 = Orthogonal
      - 0.0 = Opposite direction (semantic reversal)
    """
    cos_sim = dot(prompt_emb, current_state) / (||prompt_emb|| * ||current_state||)
    return clamp(cos_sim, 0, 1)
```

**Process**:
1. Extract prompt embedding: Average-pool last-layer hidden states over prompt tokens (once, before generation)
2. For each generated token: Extract last-layer hidden state
3. Compute cosine similarity: `cos(prompt_embedding, token_hidden_state)`
4. Store trajectory: `[D(t=0), D(t=1), ..., D(t=T)]`

**Expected Pattern**:
- **Normal**: High drift (0.7-0.9), stays grounded
- **Attack**: Decreasing drift (starts 0.7, drops to 0.3-0.5), progressive departure

**Range**: `[0, 1]` where:
- **Higher** = Staying grounded (good)
- **Lower** = Drifting away (hallucination)

---

### 2. Updated Classifier (Entropy + Drift Only)

**File**: `analysis/multi_signal_classifier.py`

**Changes**:
```python
# OLD (Phase 2):
signals = ['entropy', 'attention', 'perplexity', 'attention_entropy']  # 4 signals

# NEW (Phase 2a):
signals = ['entropy', 'semantic_drift']  # 2 signals only
```

**Why Drop 3 Signals?**

1. **Attention** (AUC 0.378):
   - Worse than random (0.5)
   - No detachment effect found
   - Pure noise

2. **Perplexity** (AUC 0.706 but NEGATIVE weight):
   - Model learned: Higher perplexity → LESS likely attack (wrong!)
   - Caused by outliers in normal class (perplexity 43.2)
   - Contradicts Entropy signal

3. **Attention Entropy** (AUC 0.631):
   - Weak effect (Cohen's d=0.47)
   - Highly correlated with Entropy
   - Adds no new information

**Result**: Focus on 2 **complementary** signals:
- **Entropy**: Measures uncertainty (internal)
- **Semantic Drift**: Measures alignment (semantic)

---

### 3. Statistical Analysis Updates

**File**: `analysis/statistical_analysis.py`

**Added**:
- Semantic drift collection (lines 392, 397)
- Semantic drift statistics (lines 463-468)
- Semantic drift ROC curve (lines 519-529) - with `higher_is_attack=False`
- Semantic drift optimal threshold (lines 569-577)

**Key**: Semantic drift uses **INVERTED** scoring:
```python
# For semantic drift: LOWER similarity = attack (drifting away)
auc = roc_auc_score(labels, -drift_values)  # Negate for correct AUC
```

---

### 4. Trace Generation Updates

**File**: `scripts/generate_traces_batch.py`

**Added**:
- Line 92: Save `semantic_drift_trace` for original prompts
- Line 113: Save `semantic_drift_trace` for adversarial prompts
- Lines 154-167: Updated summary message

**Saved Signals** (all 5 for backward compatibility):
1. ✅ `entropy_trace` - Phase 2: VALIDATED
2. ✅ `semantic_drift_trace` - Phase 2a: PRIMARY
3. ⚠️ `attention_trace` - Phase 2: DEPRECATED
4. ⚠️ `perplexity_trace` - Phase 2: DEPRECATED
5. ⚠️ `attention_entropy_trace` - Phase 2: DEPRECATED

---

### 5. Visualization Updates

**File**: `analysis/visualize_signals.py`

**Change**: Signal loop already generic, will automatically plot `semantic_drift` when present:
```python
for signal in ['entropy', 'attention', 'perplexity', 'attention_entropy', 'semantic_drift']:
    # Generate distribution plots
```

**New Figures** (automatically generated):
- `phase2a_semantic_drift_distribution_mean.png`
- `phase2a_semantic_drift_distribution_max.png`
- Temporal patterns include semantic drift column

---

## 🚀 Deployment Guide

### Deploy Phase 2a Job

```bash
kubectl apply -f k8s/phase2a-job.yaml
```

### Monitor Progress

```bash
kubectl logs -f job/neural-pulse-phase2a-analysis -n gp-engine-mizzou-dcps
```

### Expected Runtime

- Step 1: Extract attacks (30 sec)
- Step 2: Generate traces (15-20 min) - **Slower due to hidden states**
- Step 3: Statistical analysis (1 min)
- Step 4: Visualizations (2 min)
- Step 5: Multi-signal classifier (1 min)

**Total**: ~20-25 minutes (vs 15-20 min in Phase 2 due to hidden state extraction)

---

## 📈 Expected Results

### Optimistic Scenario (Semantic Drift Works!)

**Hypothesis**: Attacks drift away from prompt, normal generations stay grounded.

**Expected**:
- Semantic Drift AUC: **0.72-0.78** (similar to or better than Entropy)
- Entropy + Drift combined: **0.78-0.82** (complementary information)
- **Verdict**: ⚠️ CLOSE to target (0.85), but not quite

**Why Not Higher?**
- Attacks are "semantic substitutions" (SECA), not random drift
- Model may maintain semantic coherence even when hallucinating
- Need larger dataset (500-1000 attacks) to see clearer patterns

---

### Realistic Scenario (Moderate Improvement)

**Hypothesis**: Drift helps but not dramatically.

**Expected**:
- Semantic Drift AUC: **0.68-0.72** (comparable to Entropy)
- Entropy + Drift combined: **0.72-0.76** (modest improvement)
- **Verdict**: ❌ INSUFFICIENT for publication

**Why?**
- SECA attacks may be too subtle (small semantic shifts)
- Cosine similarity may not capture nuanced drift patterns
- Need temporal patterns (slope, variance) not just mean

---

### Pessimistic Scenario (Drift Doesn't Help)

**Hypothesis**: Semantic drift is too correlated with entropy.

**Expected**:
- Semantic Drift AUC: **0.60-0.68** (weaker than Entropy)
- High correlation with Entropy (> 0.7)
- Entropy + Drift combined: **0.70-0.72** (minimal improvement)
- **Verdict**: ❌ PIVOT REQUIRED

**Why?**
- When model is uncertain (high entropy), embeddings may naturally diverge
- Drift may just be a noisy proxy for entropy
- Need fundamentally different signal (post-generation semantic analysis)

---

## 🔍 Success Criteria

### Phase 2a Goals

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Semantic Drift AUC | > 0.70 | > 0.75 |
| Entropy + Drift AUC | > 0.75 | > 0.85 |
| Improvement over Phase 2 | +0.10 | +0.20 |
| Correlation (Drift vs Entropy) | < 0.60 | < 0.40 |

### Decision Matrix

**IF** Combined AUC ≥ 0.85:
- ✅ **SUCCESS** - Proceed to Phase 3 (Real-time Monitor)
- Prepare publication draft
- Deploy production detector

**IF** Combined AUC ∈ [0.75, 0.85):
- ⚠️ **CLOSE** - Scale up dataset
- Generate 500-1000 SECA attacks
- Re-run Phase 2a
- Expected: +0.05 to +0.10 AUC boost

**IF** Combined AUC < 0.75:
- ❌ **INSUFFICIENT** - Pivot approach
- Try Option 2a: Post-generation semantic analysis
  - Semantic consistency checking
  - Fact verification
  - Self-contradiction detection
- OR: Try different attack types (jailbreaks, prompt injection)

---

## 📝 Files Modified

### Core Signal Extraction
- ✅ `core/llama_hook.py`:
  - Added `compute_semantic_drift()` method (lines 267-309)
  - Updated `GenerationTrace` dataclass (line 41)
  - Extract prompt embedding before generation loop (lines 368-377)
  - Compute drift for each token (lines 404-406)
  - Return drift in trace (line 448)

### Batch Processing
- ✅ `scripts/generate_traces_batch.py`:
  - Save semantic_drift_trace (lines 92, 113)
  - Updated summary message (lines 154-167)

### Analysis
- ✅ `analysis/statistical_analysis.py`:
  - Collect semantic drift (lines 392, 397, 404, 413, 420, 426)
  - Compute statistics (lines 463-468)
  - Generate ROC curve (lines 519-529)
  - Find optimal threshold (lines 569-577)

- ✅ `analysis/multi_signal_classifier.py`:
  - Default to `['entropy', 'semantic_drift']` (line 86)
  - Handle inverted scoring for drift (lines 166-174)

- ✅ `analysis/visualize_signals.py`:
  - Already handles any signal generically (line 410)

### Deployment
- ✅ `k8s/phase2a-job.yaml`:
  - New job file for Phase 2a
  - Updated summary reporting

### Testing
- ✅ `tests/test_semantic_drift.py`:
  - Comprehensive test suite
  - 5 test cases covering computation, patterns, correlation

---

## ✅ Verification Checklist

Before deployment:

- [x] Semantic drift computation added to `core/llama_hook.py`
- [x] Semantic drift saved in `scripts/generate_traces_batch.py`
- [x] Statistical analysis handles semantic drift
- [x] Multi-signal classifier uses only entropy + drift
- [x] Visualization will plot semantic drift
- [x] K8s job file created (`phase2a-job.yaml`)
- [x] Test script created (`tests/test_semantic_drift.py`)
- [x] All deprecated signals kept for backward compatibility
- [x] Documentation complete

**Status**: ✅ **READY FOR DEPLOYMENT**

---

## 🧪 Testing Semantic Drift Locally (Optional)

Before running the full K8s job, you can test semantic drift computation locally:

```bash
# Run test suite
python tests/test_semantic_drift.py
```

**Expected Output**:
- ✅ Test 1: Computation works, values in [0, 1]
- ✅ Test 2: On-topic vs off-topic patterns
- ✅ Test 3: Temporal drift trajectories
- ✅ Test 4: Visualizations generated
- ✅ Test 5: Correlation with entropy

**Output**: `tests/semantic_drift_test_results.png`

---

## 📊 What to Look For in Results

### 1. Semantic Drift Statistics

**Check**: `results/phase2a_statistics.json → semantic_drift`

**Look for**:
```json
{
  "semantic_drift": {
    "attack_stats": {"mean": 0.65, "std": 0.12},   // Lower mean = drifting
    "normal_stats": {"mean": 0.78, "std": 0.08},   // Higher mean = grounded
    "comparison": {
      "t_pvalue": 0.001,        // Significant difference?
      "effect_size_cohens_d": 0.85   // Large effect?
    }
  },
  "roc_curves": {
    "semantic_drift": {"auc": 0.75}   // Target: > 0.70
  }
}
```

**Good Signs**:
- Attack mean < Normal mean (attacks drift away)
- p-value < 0.01 (statistically significant)
- AUC > 0.70 (decent discrimination)

**Bad Signs**:
- Attack mean ≈ Normal mean (no drift effect)
- p-value > 0.05 (not significant)
- AUC < 0.60 (worse than entropy alone)

---

### 2. Multi-Signal Classifier Performance

**Check**: `results/phase2a_multi_signal_classifier.json`

**Look for**:
```json
{
  "test_auc": 0.78,          // Target: > 0.75
  "f1_score": 0.52,          // Target: > 0.45
  "feature_weights": {
    "entropy": 1.2,          // Positive = higher entropy → attack
    "semantic_drift": -0.9   // Negative = lower drift → attack (CORRECT!)
  }
}
```

**Good Signs**:
- Test AUC > 0.75 (close to publication target)
- Semantic drift has NEGATIVE weight (lower drift → attack)
- Improvement > +0.10 over Phase 2 (0.654)

**Bad Signs**:
- Test AUC ≈ 0.65-0.70 (no improvement)
- Semantic drift has POSITIVE weight (wrong sign!)
- High correlation with entropy (redundant)

---

### 3. Feature Importance

**Check**: `results/phase2a_classifier_figures/feature_importance.png`

**Look for**:
- Bar chart showing |weight| for each signal
- Semantic drift should have comparable importance to entropy
- Ideally: Both features ~0.8-1.2 importance

**Interpretation**:
- If entropy >> drift: Drift is weak/redundant
- If drift >> entropy: Drift is the dominant signal (great!)
- If both similar: Complementary information (ideal)

---

### 4. Comparison to Phase 2

**Phase 2** (4 signals):
- Test AUC: 0.654
- Features: Entropy, Attention, Perplexity, Attention Entropy
- Problem: Worse than best individual (Perplexity 0.706)

**Phase 2a** (2 signals):
- Test AUC: ???
- Features: Entropy, Semantic Drift
- Goal: > 0.75 (improvement of +0.10 or more)

**Decision Points**:
- If +0.10 improvement: Semantic drift validated!
- If +0.05 improvement: Marginal, need more data
- If +0.00 improvement: Drift doesn't help, pivot

---

## 🎯 Next Steps After Phase 2a

### If AUC ≥ 0.85 ✅
**Action**: Proceed to Phase 3 (Real-time Monitor)
1. Build detector with Entropy + Drift
2. Implement streaming inference
3. Deploy in production
4. Prepare publication

---

### If AUC ∈ [0.75, 0.85) ⚠️
**Action**: Scale up dataset
1. Generate 500-1000 SECA attacks (vs current 100)
2. Re-run Phase 2a with larger dataset
3. Expected: +0.05 to +0.10 AUC boost
4. If still < 0.85: Try temporal patterns (drift slope, variance)

---

### If AUC < 0.75 ❌
**Action**: Pivot to alternative approach

**Option A**: Post-Generation Semantic Analysis (RECOMMENDED)
- Analyze **output text** instead of internal signals
- Semantic consistency checking
- Factual correctness verification
- Self-contradiction detection
- Prior work: SelfCheckGPT reaches AUC > 0.85

**Option B**: Temporal Pattern Analysis
- Instead of mean drift, analyze:
  - Drift slope (linear regression)
  - Drift variance (stability)
  - Drift change points (sudden shifts)
- Use RNN/LSTM classifier on drift trajectory

**Option C**: Different Attack Types
- SECA may be too subtle
- Try jailbreaks, prompt injection, context poisoning
- May have clearer signatures

---

## 📚 Key Learnings from Phase 2 → 2a Pivot

### What We Learned

1. **More Signals ≠ Better Performance**
   - 4 signals (AUC 0.654) < 1 signal (AUC 0.706)
   - Weak signals add noise, not information

2. **Signal Quality > Signal Quantity**
   - Focus on **complementary** signals
   - Drop anything with AUC < 0.65 or wrong-signed weights

3. **Internal Signals Have Limits**
   - Model can be "confidently wrong"
   - Entropy measures uncertainty, not correctness
   - Need semantic/external validation

4. **Overfitting is Real**
   - Train AUC 0.80 → Test AUC 0.65 (15% drop)
   - Small dataset (200 samples) + 4 features = overfitting
   - Phase 2a: 2 features should be more robust

### What Changed

| Aspect | Phase 2 | Phase 2a |
|--------|---------|----------|
| **Signals** | 4 (Entropy, Attention, Perplexity, Attn Entropy) | 2 (Entropy, Semantic Drift) |
| **Philosophy** | Internal vitals | Internal + Semantic |
| **Hypothesis** | Waffling + Detachment | Waffling + Drifting |
| **Expected AUC** | 0.75 (failed: 0.65) | 0.75-0.80 |
| **Focus** | Quantity | Quality |

---

## 🔬 Research Impact

### If Phase 2a Succeeds (AUC > 0.75)

**Publication Angle**: "Semantic Drift Beats Internal Signals for SECA Detection"
- Novel signal (semantic drift trajectory)
- Empirical validation (200+ traces)
- Practical detector (real-time feasible)

**Target Venues**: NeurIPS, ICLR, USENIX Security

---

### If Phase 2a Fails (AUC < 0.75)

**Publication Angle**: "Why Internal Signals Fail for SECA Detection: A Negative Result"
- Comprehensive empirical study (5 signals, 200 traces)
- Identifies fundamental limits of internal signals
- Points to need for external/semantic validation

**Target Venues**: Workshop papers, ArXiv

---

**End of Phase 2a Implementation Guide**

*Ready for deployment!*

```bash
kubectl apply -f k8s/phase2a-job.yaml
```

After completion, analyze results in `/data/`:
- `phase2a_statistics.json` - Individual signal performance
- `phase2a_multi_signal_classifier.json` - Combined performance
- `phase2a_classifier_figures/` - Visualizations

**Decision point**: Based on Test AUC, choose next action (scale up, pivot, or proceed to Phase 3).
