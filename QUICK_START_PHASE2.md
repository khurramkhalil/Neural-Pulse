# Quick Start: Phase 2 Analysis

**Goal**: Validate that successful SECA attacks (high adversarial score) show the "waffling signature" (high entropy + low attention).

---

## 🚀 Option 1: Full Phase 2 on Kubernetes (Recommended)

### One Command to Rule Them All

```bash
kubectl apply -f k8s/phase2-job.yaml
```

### Monitor Progress

```bash
# Watch logs
kubectl logs -f job/neural-pulse-phase2-analysis -n gp-engine-mizzou-dcps

# Check pod status
kubectl get pods -n gp-engine-mizzou-dcps | grep phase2

# If pod is running, exec into it
kubectl exec -it <pod-name> -n gp-engine-mizzou-dcps -- bash
```

### Check Outputs

```bash
# List generated files
kubectl exec -it <pod-name> -n gp-engine-mizzou-dcps -- ls -lh /data/

# View statistics
kubectl exec -it <pod-name> -n gp-engine-mizzou-dcps -- cat /data/phase2_statistics.json

# Copy files locally
kubectl cp <namespace>/<pod-name>:/data/phase2_figures ./phase2_results/
```

### Expected Runtime
- **1-2 hours total** on A100 GPU
- Trace generation: 30-60 min (200 traces)
- Statistical analysis: 5 min
- Formula mining: 10-20 min (1000+ combinations)
- Visualizations: 2-3 min

---

## 🧪 Option 2: Step-by-Step Local Execution

### Prerequisites

```bash
conda activate pt
cd /Users/khurram/Documents/Neural-Pulse
```

### Step 1: Corrected Analysis (2 min)

```bash
python analysis/analyze_pilot_results_corrected.py
```

**Output**: `results/pilot_analysis_corrected/`
- 5 PNG visualizations
- CORRECTED_ANALYSIS.md report

**View Results**:
```bash
open results/pilot_analysis_corrected/attack_effectiveness.png
open results/pilot_analysis_corrected/CORRECTED_ANALYSIS.md
```

### Step 2: Extract Top Attacks (< 1 min)

```bash
python scripts/extract_top_attacks.py \
  --input seca_attacks_pilot_100.json \
  --output datasets/top_attacks.json \
  --threshold 0.01 \
  --min-equivalence 0.85
```

**Output**: `datasets/top_attacks.json` (27 successful attacks)

### Step 3: Generate Traces (30-60 min) ⚠️ REQUIRES GPU

```bash
python scripts/generate_traces_batch.py \
  --attacks seca_attacks_pilot_100.json \
  --output datasets/pilot_traces.json \
  --validation datasets/pilot_validation.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-tokens 100
```

**Output**:
- `datasets/pilot_traces.json` (200 traces)
- `datasets/pilot_validation.json` (200 labels)

### Step 4: Statistical Analysis (5 min)

```python
python -c "
from analysis.statistical_analysis import SignalAnalyzer

analyzer = SignalAnalyzer()
results = analyzer.analyze_dataset(
    traces_path='datasets/pilot_traces.json',
    validations_path='datasets/pilot_validation.json',
    output_path='results/phase2_statistics.json'
)

print(f'Entropy AUC: {results[\"roc_curves\"][\"entropy\"][\"auc\"]:.3f}')
print(f'Attention AUC: {results[\"roc_curves\"][\"attention\"][\"auc\"]:.3f}')
"
```

**Output**: `results/phase2_statistics.json`

### Step 5: Visualizations (2-3 min)

```python
python -c "
from analysis.visualize_signals import SignalVisualizer

visualizer = SignalVisualizer()
visualizer.generate_all_visualizations(
    traces_path='datasets/pilot_traces.json',
    validation_path='datasets/pilot_validation.json',
    output_dir='results/phase2_figures',
    prefix='phase2'
)
"
```

**Output**: `results/phase2_figures/*.png` (7 visualizations)

### Step 6: Formula Mining (10-20 min)

```python
python -c "
from analysis.formula_mining import FormulaMiner

miner = FormulaMiner(metric='f1_score')
results = miner.mine_all_formulas(
    traces_path='datasets/pilot_traces.json',
    validation_path='datasets/pilot_validation.json',
    output_path='results/phase2_formula_mining.json',
    train_ratio=0.7
)

for formula_type, result in results.items():
    print(f'{formula_type}: F1={result[\"best_f1_score\"]:.3f}, Params={result[\"best_params\"]}')
"
```

**Output**: `results/phase2_formula_mining.json`

---

## 📊 Interpreting Results

### ✅ Hypothesis CONFIRMED

If you see:
- **Entropy AUC > 0.7** (clear separation between attack/normal)
- **Attention AUC > 0.7** (clear separation)
- **Formula F1 > 0.7** (good detection performance)
- **Visualizations show**: High-score attacks have high entropy, low attention

**Interpretation**: Waffling signature exists! Proceed to Phase 3 (Real-time Monitor).

### ⚠️ Hypothesis UNCLEAR

If you see:
- **AUC ~ 0.5-0.6** (weak separation)
- **Formula F1 < 0.6** (poor detection)
- **Visualizations show**: No clear pattern

**Interpretation**: Signals may not be optimal. Try:
1. Different signal definitions
2. Additional signals (perplexity, token probabilities)
3. More data (scale up Phase 1)

### ❌ Hypothesis FAILED

If you see:
- **AUC < 0.5** (no separation)
- **Formula F1 < 0.5** (random guessing)

**Interpretation**: Attacks don't work through waffling mechanism. Re-examine hypothesis.

---

## 🎯 Quick Checks

### Check if Top Attacks Were Generated

```bash
ls -lh datasets/top_attacks.json
# Should show ~60KB file with 27 attacks
```

### Check Corrected Analysis Outputs

```bash
ls -lh results/pilot_analysis_corrected/
# Should show:
#   - attack_effectiveness.png
#   - score_distribution.png
#   - top_10_attacks.png
#   - effectiveness_vs_equivalence.png
#   - CORRECTED_ANALYSIS.md
```

### Quick Stats from Corrected Analysis

```python
python -c "
import json
with open('datasets/top_attacks.json') as f:
    data = json.load(f)
print(f'Top attacks: {len(data[\"attacks\"])}')
print(f'Score range: {data[\"statistics\"][\"score_range\"][\"min\"]:.3f} - {data[\"statistics\"][\"score_range\"][\"max\"]:.3f}')
print(f'Mean score: {data[\"statistics\"][\"score_range\"][\"mean\"]:.3f}')
"
```

**Expected output**:
```
Top attacks: 27
Score range: 0.010 - 0.783
Mean score: 0.115
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'analysis'"

```bash
# Make sure you're in the project root
cd /Users/khurram/Documents/Neural-Pulse

# And running from correct conda env
conda activate pt
```

### Issue: CUDA out of memory

```bash
# Reduce batch size or trace length
python scripts/generate_traces_batch.py \
  --attacks seca_attacks_pilot_100.json \
  --output datasets/pilot_traces.json \
  --validation datasets/pilot_validation.json \
  --max-tokens 50  # Reduced from 100
```

### Issue: Oracle validator placeholder ground truths

The oracle validator currently uses placeholder ground truths. To fix:

1. Load ground truths from original MMLU dataset
2. Or: Parse ground truth from prompt answer key
3. Update `core/oracle_validator.py` line 86

---

## 📁 Expected File Structure After Phase 2

```
Neural-Pulse/
├── datasets/
│   ├── seca_attacks_pilot_100.json     (original 100 attacks)
│   ├── top_attacks.json                (27 successful attacks) ✅
│   ├── pilot_traces.json               (200 traces) ✅
│   └── pilot_validation.json           (200 labels) ✅
├── results/
│   ├── pilot_analysis_corrected/       (corrected analysis) ✅
│   │   ├── *.png                       (5 visualizations)
│   │   └── CORRECTED_ANALYSIS.md
│   ├── phase2_statistics.json          (ROC, AUC, thresholds) ✅
│   ├── phase2_formula_mining.json      (optimal parameters) ✅
│   └── phase2_figures/                 (Phase 2 visualizations) ✅
│       ├── phase2_entropy_distribution.png
│       ├── phase2_attention_distribution.png
│       ├── phase2_attack_traces_sample.png
│       ├── phase2_normal_traces_sample.png
│       ├── phase2_entropy_heatmap.png
│       ├── phase2_attention_heatmap.png
│       └── phase2_statistical_summary.png
└── k8s/
    ├── phase1-job.yaml
    └── phase2-job.yaml                 (ready to launch) ✅
```

---

## 💡 Pro Tips

1. **Start with corrected analysis first** - Takes 2 min, confirms you understand results correctly

2. **Extract top attacks before trace generation** - Only 27 attacks, easier to analyze

3. **Use Kubernetes for trace generation** - Much faster with GPU, handles long runs

4. **Save intermediate results** - Trace generation is expensive, don't lose data

5. **Check visualizations first** - Quick way to see if hypothesis holds

6. **Formula mining is optional first pass** - Can run later if statistical analysis looks good

---

## 📞 Need Help?

**Check these first**:
1. `CORRECTED_PHASE1_SUMMARY.md` - Full explanation of corrections
2. `results/pilot_analysis_corrected/CORRECTED_ANALYSIS.md` - Detailed analysis
3. `docs/PHASE2_COMPLETE.md` - Phase 2 implementation details

**Common Questions**:

Q: "Why are low scores bad?"
A: Low score = model only X% confident in WRONG answer = attack failed

Q: "Should I re-run Phase 1?"
A: No! You have 27 successful attacks (score > 0.01). That's plenty for validation.

Q: "What if AUC < 0.7?"
A: Could mean: (1) Need more data, (2) Different signals needed, (3) Hypothesis wrong

---

**Quick Start Complete!** 🎉

Choose your path:
- **Fast track**: `kubectl apply -f k8s/phase2-job.yaml`
- **Step-by-step**: Follow Option 2 above

Good luck! 🚀
