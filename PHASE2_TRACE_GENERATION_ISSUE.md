# Phase 2: Trace Generation Issue Report

**Date**: 2026-01-10
**Status**: ⚠️ **CRITICAL ISSUE FOUND**

---

## 🔴 **Problem: Entropy Computation Returns NaN**

### Issue Description

When generating traces using `LlamaSignalHook`, all entropy values are **NaN** (Not a Number). This prevents Phase 2 analysis from working correctly.

### Evidence

From the generated traces (`results/pilot_traces.json`):
```json
{
  "entropy_trace": [nan, nan, nan, nan, nan, ...],
  "attention_trace": [27.85, 30.45, 30.48, 31.07, ...]
}
```

- **All 200 traces** have NaN entropy values
- Attention values are present but unnormalized (range 27-31 instead of 0-1)

### Root Cause

The entropy computation in `core/llama_hook.py` has a bug. Likely causes:
1. Division by zero or log(0) in entropy calculation
2. Incorrect tensor handling
3. Missing softmax normalization before entropy computation

### Impact on Analysis

Since all entropy values are identical (replaced with 2.0 placeholder):
- **Entropy AUC = 0.500** (random guessing - no separation)
- Cannot validate "waffling signature" hypothesis
- Statistical analysis shows no entropy difference between attack/normal traces

**Attention** shows weak separation (AUC = 0.564), but not strong enough.

---

## 🔧 **Temporary Fix Applied**

Created `scripts/fix_traces.py` to:
1. Replace NaN entropy values with placeholder (2.0)
2. Normalize attention values to [0, 1] range

**Output**: `results/pilot_traces_fixed.json`

**Limitation**: This doesn't give us real entropy values, just makes analysis runnable.

---

## 📊 **Analysis Results (With Placeholder Entropy)**

### Statistical Analysis

```
Dataset:
  - Total traces: 200
  - Attack traces: 33 (high adversarial score)
  - Normal traces: 167 (low adversarial score)

ROC Analysis:
  - Entropy AUC: 0.500 (random - all values same)
  - Attention AUC: 0.564 (weak separation)

Optimal Thresholds:
  - Entropy threshold: 2.000 (meaningless - all same)
  - Attention threshold: 0.459
    F1: 0.301 (poor performance)
```

### Interpretation

❌ **Cannot validate hypothesis** with current data:
- Entropy signal is broken (NaN values)
- Attention shows weak effect but not strong enough
- Need to fix entropy computation to proceed

---

## 🛠️ **How to Fix**

### Option 1: Fix LlamaHook Entropy Computation (Recommended)

**File to fix**: `core/llama_hook.py`

**Current code** (lines 90-110):
```python
def compute_entropy(self, logits: torch.Tensor) -> float:
    """Compute Shannon entropy"""
    # BUG IS SOMEWHERE HERE
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)  # Add epsilon for stability
    entropy = -torch.sum(probs * log_probs).item()
    return entropy
```

**Possible issues**:
1. Logits might be wrong shape
2. Softmax dimension incorrect
3. NaN propagating from earlier computation

**Debug steps**:
1. Add logging to print logits shape and values
2. Check for NaN in logits before softmax
3. Verify softmax output sums to 1.0
4. Test entropy computation on simple example

### Option 2: Use Alternative Entropy Metric

Instead of Shannon entropy, use:
- **Perplexity**: `exp(-log_prob)`
- **Token confidence**: `max(softmax(logits))`
- **Negative log probability**: `-log(prob[predicted_token])`

### Option 3: Generate Traces with Fixed Code

Re-run trace generation after fixing LlamaHook:
```bash
python scripts/generate_traces_batch.py \
  --attacks seca_attacks_pilot_100.json \
  --output datasets/pilot_traces_v2.json \
  --validation datasets/pilot_validation_v2.json
```

---

## 📋 **Immediate Action Items**

### Priority 1: Fix Entropy Computation ⚠️

1. **Debug `core/llama_hook.py`**:
   ```bash
   # Add debug logging
   python -c "
   from core.llama_hook import LlamaSignalHook
   import torch

   hook = LlamaSignalHook()

   # Test entropy on simple case
   logits = torch.randn(50257)  # Llama vocab size
   entropy = hook.compute_entropy(logits)
   print(f'Test entropy: {entropy}')
   print(f'Is NaN: {torch.isnan(torch.tensor(entropy))}')
   "
   ```

2. **Identify root cause**:
   - Check if logits contain NaN
   - Verify softmax computation
   - Test with different input sizes

3. **Fix the bug**:
   - Add proper NaN handling
   - Ensure numerical stability
   - Add validation checks

### Priority 2: Regenerate Traces

Once fixed:
```bash
# Delete old traces
rm results/pilot_traces.json

# Regenerate with fixed code
python scripts/generate_traces_batch.py \
  --attacks seca_attacks_pilot_100.json \
  --output results/pilot_traces_v2.json \
  --validation results/pilot_validation_v2.json
```

### Priority 3: Re-run Phase 2 Analysis

With valid entropy values:
```bash
python scripts/run_phase2_analysis.py \
  --traces results/pilot_traces_v2.json \
  --validations results/pilot_validation_v2.json \
  --output-dir results/phase2_final
```

---

## 🔍 **Diagnostic Commands**

### Check if LlamaHook works

```bash
python -c "
from core.llama_hook import LlamaSignalHook

hook = LlamaSignalHook()
trace = hook.generate_with_signals(
    prompt='What is 2+2?',
    max_new_tokens=10
)

print('Generated:', trace.generated_text)
print('Entropy trace:', trace.entropy_trace[:5])
print('Has NaN:', any(float('nan') == x or x != x for x in trace.entropy_trace))
"
```

### Inspect raw trace data

```bash
python -c "
import json

with open('results/pilot_traces.json') as f:
    traces = json.load(f)

trace = traces[0]
print('Entropy values (first 10):', trace['entropy_trace'][:10])
print('Attention values (first 10):', trace['attention_trace'][:10])
"
```

---

## 📖 **What We Learned**

### Attention Signal Works
- Values generated successfully (though needed normalization)
- Shows weak separation (AUC 0.564)
- May not be strong enough alone

### Entropy Signal Critical
- Currently broken (NaN values)
- Once fixed, should show stronger separation
- Combined with attention, should validate hypothesis

### Data Quality Validation Needed
- Should check for NaN values immediately after generation
- Add validation step in trace generation pipeline
- Fail fast if data is invalid

---

## 🎯 **Next Steps**

1. ✅ **Fix entropy computation** in `core/llama_hook.py`
2. ✅ **Regenerate traces** with fixed code
3. ✅ **Re-run Phase 2 analysis** with valid data
4. ⏳ **Validate waffling signature** (can't do until #1-3 done)
5. ⏳ **Proceed to Phase 3** if validation successful

---

## 📞 **Current Status**

**Phase 1**: ✅ Complete (100 attacks, 27 successful)
**Phase 2**: ⚠️ **BLOCKED** on entropy computation bug
**Phase 3**: ⏸️ Waiting for Phase 2 validation

**Blocker**: `core/llama_hook.py` entropy computation returns NaN

**Solution**: Debug and fix `compute_entropy()` method, then regenerate all traces

---

## 📝 **Files for Reference**

- **Bug location**: `core/llama_hook.py` lines 90-110
- **Broken traces**: `results/pilot_traces.json`
- **Fixed (placeholder) traces**: `results/pilot_traces_fixed.json`
- **Fix script**: `scripts/fix_traces.py`
- **Issue report**: This document

---

**Priority**: **HIGH** - Cannot proceed to Phase 3 without fixing this

**Assignee**: Need to debug `core/llama_hook.py`

**ETA**: Once debugged, ~1-2 hours to regenerate traces and complete Phase 2

---

**End of Issue Report**
