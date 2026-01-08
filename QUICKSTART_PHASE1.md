# Neural Pulse - Phase 1 Quick Start Guide

## Prerequisites

### 1. System Requirements
- **Python**: 3.9+
- **GPU**: NVIDIA A100 (recommended) or CPU (slower)
- **RAM**: 32GB+ for Llama-3-8B
- **Storage**: 20GB+ for models

### 2. Environment Setup

```bash
# Clone repository
cd /Users/khurram/Documents/Neural-Pulse

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. API Keys and Authentication

```bash
# OpenAI API Key (for SECA generation)
export OPENAI_API_KEY="sk-your-key-here"

# HuggingFace Token (for Llama-3-8B access)
export HF_TOKEN="hf_your-token-here"
huggingface-cli login
```

### 4. Model Downloads
Models will auto-download on first use:
- **Llama-3-8B-Instruct**: ~16GB (meta-llama/Llama-3.1-8B-Instruct)
- **DeBERTa-large**: ~1.5GB (microsoft/deberta-large-mnli)

---

## Phase 1 Execution (Week 1-3)

### Step 1: Run Unit Tests (Validation)

```bash
# Run all tests
python -m pytest tests/unit/ -v

# Run specific component tests
python -m pytest tests/unit/test_oracle_validator.py -v
python -m pytest tests/unit/test_seca_generator.py -v
python -m pytest tests/unit/test_llama_hook.py -v

# Run with coverage report
python -m pytest tests/unit/ --cov=datasets --cov=core --cov-report=html
# Open htmlcov/index.html to view coverage
```

**Expected Output**:
- ✅ All tests pass (78 tests total)
- ⚠️ Some tests may be slow (model loading)
- 📊 Coverage should be >85%

---

### Step 2: Prepare MMLU Dataset

First, obtain the filtered MMLU dataset from SECA repository:

```bash
# Clone SECA repository
cd /tmp
git clone https://github.com/Buyun-Liang/SECA.git
cd SECA

# Copy filtered MMLU to Neural-Pulse datasets/
cp filtered_mmlu.json /Users/khurram/Documents/Neural-Pulse/datasets/

cd /Users/khurram/Documents/Neural-Pulse
```

**Filtered MMLU format**:
```json
[
  {
    "prompt": "What is the capital of France? (A) London (B) Paris (C) Berlin (D) Rome",
    "answer": "B",
    "subject": "geography",
    "id": 0
  },
  ...
]
```

---

### Step 3: Generate SECA Attacks (100 pilot)

**Cost**: ~$0.60 for 100 attacks

```bash
python datasets/generate_seca_attacks.py \
  --source datasets/filtered_mmlu.json \
  --output datasets/seca_attacks_pilot_100.json \
  --num_attacks 100 \
  --proposer gpt-4o-mini-2024-07-18 \
  --max_iterations 30 \
  --device cuda  # or cpu
```

**Expected Runtime**: ~45 minutes (depends on API latency)

**Output** (`datasets/seca_attacks_pilot_100.json`):
```json
{
  "generator": {
    "proposer": "gpt-4o-mini-2024-07-18",
    "n_candidates": 3,
    "m_rephrasings": 3,
    "max_iterations": 30
  },
  "attacks": [
    {
      "original_prompt": "What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6",
      "adversarial_prompt": "Calculate the sum of two and two. (A) 3 (B) 4 (C) 5 (D) 6",
      "iterations": 23,
      "success": true,
      "adversarial_score": 0.72,
      "equivalence_score": 0.91
    },
    ...
  ],
  "statistics": {
    "total": 100,
    "successful": 87,
    "avg_iterations": 23.4,
    "avg_adversarial_score": 0.68
  }
}
```

---

### Step 4: Extract Signal Traces

#### 4a. Extract from SECA attacks (100 traces)

```bash
python core/llama_hook.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompts datasets/seca_attacks_pilot_100.json \
  --output datasets/traces_seca_100.json \
  --max_tokens 100 \
  --temperature 1.0 \
  --device cuda  # or cpu
```

**Expected Runtime**: ~1 hour on A100, ~4 hours on CPU

#### 4b. Extract from MMLU clean prompts (100 traces)

```bash
# First, create a subset of MMLU prompts
python -c "
import json
with open('datasets/filtered_mmlu.json', 'r') as f:
    data = json.load(f)
with open('datasets/mmlu_pilot_100.json', 'w') as f:
    json.dump(data[:100], f, indent=2)
"

# Extract traces
python core/llama_hook.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompts datasets/mmlu_pilot_100.json \
  --output datasets/traces_mmlu_100.json \
  --max_tokens 100 \
  --temperature 1.0 \
  --device cuda
```

**Output** (`datasets/traces_seca_100.json`):
```json
{
  "traces": [
    {
      "prompt": "Calculate the sum of two and two. (A) 3 (B) 4 (C) 5 (D) 6",
      "generated_text": "The answer is (A) because...",
      "generated_tokens": ["The", " answer", " is", ...],
      "entropy_trace": [2.3, 2.8, 3.1, 2.9, ...],  // H(t) for each token
      "attention_trace": [0.45, 0.32, 0.28, 0.25, ...],  // A(t) for each token
      "signals": {
        "avg_entropy": 2.87,
        "max_entropy": 3.42,
        "avg_attention": 0.31,
        "min_attention": 0.21
      }
    },
    ...
  ]
}
```

---

### Step 5: Validate Traces with Oracle

#### 5a. Combine all traces

```bash
python -c "
import json

# Load SECA traces
with open('datasets/traces_seca_100.json', 'r') as f:
    seca_data = json.load(f)
    seca_traces = seca_data['traces']

# Load MMLU traces
with open('datasets/traces_mmlu_100.json', 'r') as f:
    mmlu_data = json.load(f)
    mmlu_traces = mmlu_data['traces']

# Combine
all_traces = seca_traces + mmlu_traces

# Save
with open('datasets/all_traces_200.json', 'w') as f:
    json.dump(all_traces, f, indent=2)

print(f'Combined {len(all_traces)} traces')
"
```

#### 5b. Create ground truth file

```bash
python -c "
import json

# Load SECA attacks
with open('datasets/seca_attacks_pilot_100.json', 'r') as f:
    seca = json.load(f)
    seca_gt = [{'answer': a.get('answer', 'A')} for a in seca['attacks']]

# Load MMLU prompts
with open('datasets/mmlu_pilot_100.json', 'r') as f:
    mmlu = json.load(f)
    mmlu_gt = [{'answer': item['answer']} for item in mmlu]

# Combine
all_gt = seca_gt + mmlu_gt

# Save
with open('datasets/ground_truth_200.json', 'w') as f:
    json.dump(all_gt, f, indent=2)

print(f'Created ground truth for {len(all_gt)} prompts')
"
```

#### 5c. Run oracle validation

```bash
python datasets/oracle_validator.py \
  --traces datasets/all_traces_200.json \
  --ground_truth datasets/ground_truth_200.json \
  --output datasets/validated_traces_200.json \
  --confidence_threshold 0.9 \
  --device cuda  # or cpu
```

**Expected Runtime**: ~10 minutes on A100

**Output** (`datasets/validated_traces_200.json`):
```json
{
  "validation_results": [
    {
      "trace_id": 0,
      "is_hallucination": true,
      "hallucination_type": "factuality",
      "confidence": 0.94,
      "explanation": "Answer 'A' is wrong (expected 'B'), but explanation is consistent",
      "nli_label": "neutral",
      "nli_score": 0.87
    },
    ...
  ],
  "statistics": {
    "total_traces": 200,
    "hallucination_count": 92,
    "hallucination_rate": 0.46,
    "factuality_errors": 78,
    "faithfulness_errors": 14,
    "high_confidence_predictions": 187,
    "avg_confidence": 0.92,
    "low_confidence_count": 13
  }
}
```

---

### Step 6: Analyze Results

```bash
python -c "
import json

# Load validation results
with open('datasets/validated_traces_200.json', 'r') as f:
    data = json.load(f)

stats = data['statistics']
results = data['validation_results']

# Split by source (first 100 = SECA, last 100 = MMLU)
seca_results = results[:100]
mmlu_results = results[100:]

seca_hallucinations = sum(1 for r in seca_results if r['is_hallucination'])
mmlu_hallucinations = sum(1 for r in mmlu_results if r['is_hallucination'])

print('=== Validation Results Summary ===')
print(f'Total traces: {stats[\"total_traces\"]}')
print(f'Overall hallucination rate: {stats[\"hallucination_rate\"]:.1%}')
print()
print('SECA Attacks:')
print(f'  Hallucinations: {seca_hallucinations}/100 ({seca_hallucinations}%)')
print(f'  Expected: ~85%')
print()
print('MMLU Clean:')
print(f'  Hallucinations: {mmlu_hallucinations}/100 ({mmlu_hallucinations}%)')
print(f'  Expected: ~5%')
print()
print('Hallucination Types:')
print(f'  Factuality errors: {stats[\"factuality_errors\"]}')
print(f'  Faithfulness errors: {stats[\"faithfulness_errors\"]}')
print()
print('Confidence:')
print(f'  High confidence (>0.9): {stats[\"high_confidence_predictions\"]}/{stats[\"total_traces\"]}')
print(f'  Average confidence: {stats[\"avg_confidence\"]:.3f}')
"
```

**Expected Output**:
```
=== Validation Results Summary ===
Total traces: 200
Overall hallucination rate: 46.0%

SECA Attacks:
  Hallucinations: 87/100 (87%)
  Expected: ~85%

MMLU Clean:
  Hallucinations: 5/100 (5%)
  Expected: ~5%

Hallucination Types:
  Factuality errors: 78
  Faithfulness errors: 14

Confidence:
  High confidence (>0.9): 187/200
  Average confidence: 0.920
```

---

## Scaling to Full Dataset (Week 2-3)

Once pilot (200 traces) is validated, scale to full dataset:

```bash
# Generate 1000 SECA attacks
python datasets/generate_seca_attacks.py \
  --source datasets/filtered_mmlu.json \
  --output datasets/seca_attacks_1000.json \
  --num_attacks 1000 \
  --max_iterations 30

# Extract 1000 SECA traces
python core/llama_hook.py \
  --prompts datasets/seca_attacks_1000.json \
  --output datasets/traces_seca_1000.json \
  --max_tokens 100

# Extract 1000 MMLU traces
python core/llama_hook.py \
  --prompts datasets/filtered_mmlu.json \
  --output datasets/traces_mmlu_1000.json \
  --max_tokens 100

# Validate all 2000 traces
python datasets/oracle_validator.py \
  --traces datasets/all_traces_2000.json \
  --ground_truth datasets/ground_truth_2000.json \
  --output datasets/validated_traces_2000.json
```

**Total Cost**: $6 (SECA generation only)
**Total Runtime**: ~12 hours on A100 (mostly trace extraction)

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Use CPU or reduce batch size
```bash
# Use CPU instead
--device cpu

# Or use smaller model for testing
# (Note: Not recommended for final dataset)
```

### Issue: OpenAI API rate limit
**Solution**: Add delay between requests or use Tier 2+ account
```python
# In generate_seca_attacks.py, add:
import time
time.sleep(1)  # After each API call
```

### Issue: HuggingFace model access denied
**Solution**: Accept Llama-3 license on HuggingFace
```bash
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Click "Agree and access repository"
huggingface-cli login
```

### Issue: Tests fail with model errors
**Solution**: Ensure models are downloaded first
```bash
# Pre-download models
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('Models downloaded successfully')
"
```

---

## Next Steps: Phase 2

After Phase 1 completion:
1. ✅ 2000 validated traces with signals
2. ✅ ~900 hallucination traces, ~1100 correct traces
3. ✅ Signal distributions documented

**Proceed to Phase 2**: Diagnosis & Formula Mining
- Visualize signal patterns (entropy vs attention)
- Derive STL formula thresholds (θ_H, θ_A, T)
- Implement STL formulas φ₁, φ₂, φ₃

See `MASTER_PLAN.md` for Phase 2 details.

---

## Quick Reference

### Key Files
- **Oracle**: `datasets/oracle_validator.py`
- **SECA Gen**: `datasets/generate_seca_attacks.py`
- **Signals**: `core/llama_hook.py`
- **Tests**: `tests/unit/test_*.py`
- **Config**: `.neuralpulse/config.json`

### Key Commands
```bash
# Run tests
pytest tests/unit/ -v

# Generate attacks
python datasets/generate_seca_attacks.py --source ... --output ...

# Extract signals
python core/llama_hook.py --prompts ... --output ...

# Validate traces
python datasets/oracle_validator.py --traces ... --ground_truth ... --output ...
```

### Expected Metrics
- SECA → Hallucination: ~85%
- MMLU → Hallucination: ~5%
- Oracle confidence: >0.9 for 90%+ traces
- Total cost: $6 for full dataset

---

**Happy experimenting!** 🚀
