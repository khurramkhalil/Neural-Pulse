# Neural Pulse: Research & Implementation Master Plan
**Runtime Verification Framework for Adversarial Defense Against SECA Attacks**

**Target Venues:** NeurIPS, ICLR, CCS
**Defense Method:** Signal Temporal Logic (STL) Runtime Verification
**Base Threat Model:** SECA (Semantically Equivalent and Coherent Attacks - NeurIPS 2025)

---

## Executive Summary

**Research Hypothesis:** SECA attacks force LLMs into a state of "cognitive dissonance" or "waffling" before hallucination, producing a distinct temporal signature detectable via runtime monitoring of generation trajectories.

**Key Innovation:** Unlike static input/output filters, Neural Pulse monitors the *process* of generation in real-time using STL formulas over temporal signals (Entropy, Attention Dispersion).

**Success Criteria:**
- Detection Rate (TPR) > 85% on SECA attacks
- False Positive Rate < 10% on normal MMLU questions
- Runtime overhead < 20% compared to baseline inference
- Publication-ready plots, tables, and rigorous ablation studies

---

## Phase 1: The "Oscilloscope" (Data & Signal Extraction)

### 1.1 Environment Setup (Week 1)

**Infrastructure:**
- Nautilus K8s cluster (namespace: `gp-engine-mizzou-dcps`)
- GPU: NVIDIA A100-80GB
- Container: `khurramkhalil/gpa-emergency:latest`

**Dependencies:**
```bash
# Core ML Stack
pip install torch transformers accelerate
# STL Library
pip install rtamt
# Data & Analysis
pip install datasets numpy pandas matplotlib seaborn
# SECA Repository
git clone https://github.com/Buyun-Liang/SECA
```

**K8s Job Template:** (See `infrastructure/k8s/neuralpulse-job.yaml`)
- Request: 1x A100, 32Gi RAM, 8 CPUs
- Toleration: `nautilus.io/reservation=mizzou`
- Node Affinity: Target A100 nodes

### 1.2 Dataset Preparation (Week 1-2)

**Attack Dataset:**
```python
# datasets/seca_loader.py
class SECADataset:
    """
    Load SECA adversarial prompts
    Source: https://github.com/Buyun-Liang/SECA
    """
    def __init__(self, split='test'):
        # Load SECA-generated adversarial MMLU variants
        # Structure: {original_prompt, adversarial_prompt, target_token, ground_truth}
```

**Normal Dataset:**
```python
# datasets/mmlu_loader.py
class MMLUDataset:
    """
    Load clean MMLU multiple-choice questions
    Filter: Only questions where target model answers correctly
    """
    def __init__(self, subjects=['elementary_mathematics', 'physics', ...]):
        # Load MMLU subset across 16 subjects
        # Pre-filter: model.predict(prompt) == ground_truth
```

**Data Split Strategy:**
- Train/Val/Test: 60/20/20 split
- Stratified by MMLU subject to ensure balanced representation
- Minimum 50 samples per subject in test set

### 1.3 LlamaHook Implementation (Week 2-3)

**Design Pattern:** Hook into HuggingFace `model.generate()` to extract signals at every token step.

```python
# core/llama_hook.py
class LlamaSignalHook:
    """
    Runtime signal extraction during generation.
    Extracts:
      - Entropy: H(t) = -Σ p(x_i) log p(x_i) at each token t
      - Attention Dispersion: A(t) = attention_to_context / total_attention
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.signals = {'entropy': [], 'attention': []}

    def register_hooks(self):
        """
        Register forward hooks on:
        1. Final layer output (for logits -> entropy)
        2. Attention layers (for attention weights)
        """

    def compute_entropy(self, logits):
        """
        H(t) = -Σ p_i * log(p_i)
        where p_i = softmax(logits)
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()

    def compute_attention_dispersion(self, attention_weights, context_length, generated_length):
        """
        A(t) = mean_attention_to_context / mean_attention_to_generated

        attention_weights: [batch, heads, seq_len, seq_len]
        context: tokens [0:context_length]
        generated: tokens [context_length:]
        """
        context_attn = attention_weights[..., :context_length].mean()
        generated_attn = attention_weights[..., context_length:].mean()
        return (context_attn / (generated_attn + 1e-10)).item()

    def generate_with_signals(self, prompt, max_new_tokens=100):
        """
        Wrapper around model.generate() that logs signals at each step.
        Returns: (generated_text, signal_trace)
        """
```

**Expected Output Format:**
```json
{
  "prompt": "If doubling the value of p results in 24, what is p?",
  "generated_tokens": ["The", "answer", "is", "B", "."],
  "signals": {
    "entropy": [0.45, 0.52, 0.61, 0.58, 0.32],
    "attention_dispersion": [0.85, 0.78, 0.65, 0.52, 0.48]
  },
  "label": "attack"
}
```

### 1.4 Trace Generation Pipeline (Week 3-4)

```python
# experiments/01_extract_traces.py
"""
Generate signal traces for entire dataset.
Output: traces/{attack|normal}/*.json
"""

def extract_traces(model, dataset, output_dir):
    hook = LlamaSignalHook(model, tokenizer)

    for sample in tqdm(dataset):
        trace = hook.generate_with_signals(sample['prompt'])

        # Save trace
        trace_id = f"{sample['subject']}_{sample['id']}"
        with open(f"{output_dir}/{sample['label']}/{trace_id}.json", 'w') as f:
            json.dump(trace, f)
```

**Deliverables (Phase 1):**
- [ ] Fully functional `LlamaHook` with entropy + attention extraction
- [ ] 1000+ attack traces (SECA prompts)
- [ ] 1000+ normal traces (MMLU prompts)
- [ ] Data directory structure: `data/traces/{attack|normal}/`
- [ ] Sanity check: Verify signals are logged correctly (manual inspection)

---

## Phase 2: Diagnosis & Formula Mining

### 2.1 Visualization & Signature Discovery (Week 4-5)

**Goal:** Identify the "Waffle Signature" - the temporal pattern that distinguishes attacks from normal.

```python
# analysis/visualize_traces.py
"""
Generate publication-quality plots comparing attack vs normal traces.
"""

def plot_entropy_over_time(attack_traces, normal_traces):
    """
    Figure 1: Entropy H(t) over time
    - X-axis: Token position (t = 0, 1, 2, ...)
    - Y-axis: Entropy value
    - Red lines: Attack traces (with confidence band)
    - Blue lines: Normal traces (with confidence band)
    """

def plot_attention_over_time(attack_traces, normal_traces):
    """
    Figure 2: Attention Dispersion A(t) over time
    Similar layout to Figure 1
    """

def plot_phase_space(attack_traces, normal_traces):
    """
    Figure 3: 2D Phase Space (Entropy vs Attention)
    - X-axis: Entropy
    - Y-axis: Attention Dispersion
    - Scatter plot with attack (red) vs normal (blue) points
    - Goal: Show separability
    """
```

**Expected Observations:**
1. **Attack Signature (Hypothesis):**
   - High entropy plateau (H > 0.6) for extended period (5+ tokens)
   - Low context attention (A < 0.4) during generation
   - Correlation: High H + Low A = Attack

2. **Normal Signature:**
   - Entropy drops after initial uncertainty (H decreases over time)
   - Stable context attention (A > 0.6)

### 2.2 Statistical Analysis (Week 5-6)

```python
# analysis/statistical_tests.py
"""
Rigorous statistical validation of hypothesized differences.
"""

def compute_signature_statistics(traces):
    """
    For each trace, compute:
    - max_entropy: max_t H(t)
    - mean_entropy: mean_t H(t)
    - entropy_duration_above_threshold: # tokens where H(t) > θ_H
    - min_attention: min_t A(t)
    - mean_attention: mean_t A(t)
    """

def compare_distributions(attack_stats, normal_stats):
    """
    Statistical tests:
    - Welch's t-test: Compare means of attack vs normal
    - Kolmogorov-Smirnov test: Compare distributions
    - Effect size: Cohen's d

    Report p-values and effect sizes for each signal.
    """
```

**Output Table (for paper):**

| Signal Feature       | Attack (mean ± std) | Normal (mean ± std) | p-value | Cohen's d |
|----------------------|---------------------|---------------------|---------|-----------|
| Max Entropy          | 0.72 ± 0.08         | 0.48 ± 0.12         | <0.001  | 2.35      |
| Mean Entropy         | 0.61 ± 0.07         | 0.42 ± 0.09         | <0.001  | 2.18      |
| Min Attention Disp.  | 0.38 ± 0.11         | 0.62 ± 0.08         | <0.001  | -2.54     |
| Mean Attention Disp. | 0.45 ± 0.09         | 0.71 ± 0.06         | <0.001  | -3.12     |

### 2.3 STL Formula Derivation (Week 6-7)

**STL Background (using rtamt):**
```python
import rtamt

# STL Operators:
# - Always[a,b](φ): φ holds for all t in [a, b]
# - Eventually[a,b](φ): φ holds for some t in [a, b]
# - Until(φ, ψ): φ holds until ψ becomes true
# - Not(φ), And(φ, ψ), Or(φ, ψ)
```

**Candidate Formulas:**

```python
# formulas/stl_specifications.py
"""
STL formulas for attack detection.
Thresholds derived from Phase 2.1-2.2 analysis.
"""

def formula_1_waffling(θ_H=0.6, duration=5):
    """
    Formula 1: Prolonged High Entropy (Waffling)

    φ_1 = Eventually[0,T](Always[t,t+duration](H(t) > θ_H))

    Meaning: At some point during generation, entropy stays high for 'duration' tokens.
    """
    spec = f"""
    specification waffle {{
        input signal H: float
        output signal attack: bool

        attack = eventually[0:T](always[0:{duration}](H > {θ_H}))
    }}
    """
    return rtamt.STLDiscreteTimeSpecification()

def formula_2_context_detachment(θ_A=0.4, duration=3):
    """
    Formula 2: Context Detachment

    φ_2 = Eventually[0,T](Always[t,t+duration](A(t) < θ_A))

    Meaning: Model detaches from context (low attention to context) for sustained period.
    """

def formula_3_combined(θ_H=0.6, θ_A=0.4, overlap=3):
    """
    Formula 3: Combined Signature (Strongest Detector)

    φ_3 = Eventually[0,T](Always[t,t+overlap](H(t) > θ_H AND A(t) < θ_A))

    Meaning: Both waffling AND context detachment occur simultaneously.
    """
```

**Threshold Tuning:**
```python
# experiments/02_tune_thresholds.py
"""
Grid search over threshold parameters to maximize F1 score on validation set.
"""

def grid_search_thresholds(val_traces, formula_template):
    best_f1 = 0
    best_params = {}

    for θ_H in np.linspace(0.4, 0.8, 20):
        for θ_A in np.linspace(0.2, 0.6, 20):
            for duration in range(3, 10):
                formula = formula_template(θ_H, θ_A, duration)
                f1 = evaluate_formula(formula, val_traces)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'θ_H': θ_H, 'θ_A': θ_A, 'duration': duration}

    return best_params
```

**Deliverables (Phase 2):**
- [ ] Publication plots showing "Waffling Signature" (Entropy/Attention over time)
- [ ] Statistical analysis table (p-values, effect sizes)
- [ ] 3 STL formulas with optimized thresholds
- [ ] LaTeX figure captions ready for paper

---

## Phase 3: The STL Monitor (The Defense)

### 3.1 NeuralPulse Monitor Design (Week 7-8)

```python
# core/neuralpulse_monitor.py
"""
Runtime defense system that monitors generation and blocks attacks.
"""

class NeuralPulseMonitor:
    """
    Real-time STL monitor for attack detection.

    Usage:
        monitor = NeuralPulseMonitor(model, stl_formula)
        safe_output, is_attack = monitor.safe_generate(prompt)
    """

    def __init__(self, model, tokenizer, stl_formula, max_new_tokens=100):
        self.model = model
        self.hook = LlamaSignalHook(model, tokenizer)
        self.stl_formula = stl_formula
        self.max_new_tokens = max_new_tokens

    def safe_generate(self, prompt):
        """
        Generate with runtime monitoring.

        At each token t:
          1. Extract signals H(t), A(t)
          2. Check STL formula φ(H[0:t], A[0:t])
          3. If φ evaluates to TRUE (attack detected):
               - Stop generation
               - Return: (None, is_attack=True)
          4. If generation completes without violation:
               - Return: (generated_text, is_attack=False)
        """

        self.hook.register_hooks()
        signal_buffer = {'entropy': [], 'attention': []}

        # Token-by-token generation
        for t in range(self.max_new_tokens):
            # Generate next token
            token = self.model.generate_next_token()

            # Extract signals
            H_t = self.hook.compute_entropy(logits)
            A_t = self.hook.compute_attention_dispersion(attention, ...)

            signal_buffer['entropy'].append(H_t)
            signal_buffer['attention'].append(A_t)

            # Evaluate STL formula
            is_attack = self.stl_formula.evaluate(signal_buffer)

            if is_attack:
                print(f"[BLOCKED] Attack detected at token {t}")
                return None, True

        # No violation detected
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text, False
```

### 3.2 Evaluation Metrics (Week 8-9)

```python
# evaluation/metrics.py
"""
Defense evaluation metrics.
"""

def evaluate_defense(monitor, test_dataset):
    """
    Compute:
    - True Positive Rate (TPR): % of SECA attacks correctly blocked
    - False Positive Rate (FPR): % of normal prompts incorrectly blocked
    - F1 Score: Harmonic mean of precision and recall
    - Latency Overhead: Average slowdown compared to baseline
    """

    TP, FP, TN, FN = 0, 0, 0, 0
    latencies_baseline = []
    latencies_monitor = []

    for sample in test_dataset:
        # Baseline (no defense)
        t_start = time.time()
        baseline_output = model.generate(sample['prompt'])
        latencies_baseline.append(time.time() - t_start)

        # With NeuralPulse monitor
        t_start = time.time()
        safe_output, is_blocked = monitor.safe_generate(sample['prompt'])
        latencies_monitor.append(time.time() - t_start)

        # Update confusion matrix
        if sample['label'] == 'attack':
            if is_blocked:
                TP += 1  # Correctly blocked attack
            else:
                FN += 1  # Missed attack
        else:  # normal
            if is_blocked:
                FP += 1  # False alarm
            else:
                TN += 1  # Correctly allowed

    # Compute metrics
    TPR = TP / (TP + FN)  # Detection Rate
    FPR = FP / (FP + TN)  # False Alarm Rate
    precision = TP / (TP + FP)
    recall = TPR
    f1 = 2 * (precision * recall) / (precision + recall)

    overhead = (np.mean(latencies_monitor) / np.mean(latencies_baseline) - 1) * 100

    return {
        'TPR': TPR,
        'FPR': FPR,
        'F1': f1,
        'latency_overhead_%': overhead
    }
```

**Target Performance:**
- **TPR (Detection Rate):** > 85%
- **FPR (False Positive Rate):** < 10%
- **F1 Score:** > 0.85
- **Latency Overhead:** < 20%

### 3.3 Baseline Comparisons (Week 9-10)

**Baseline 1: Static Input Filter (Perplexity)**
```python
# baselines/perplexity_filter.py
"""
Reject prompts with perplexity > threshold.
(Common but weak defense - SECA bypasses this)
"""
def perplexity_filter(prompt, threshold=50):
    ppl = compute_perplexity(prompt)
    if ppl > threshold:
        return None, True  # Blocked
    else:
        return model.generate(prompt), False
```

**Baseline 2: Output Checker (Post-hoc)**
```python
# baselines/output_checker.py
"""
Generate first, then check output for hallucination markers.
(Too late - damage already done)
"""
def output_checker(prompt):
    output = model.generate(prompt)
    is_hallucination = check_factuality(output)
    return output, is_hallucination
```

**Comparison Table (for paper):**

| Method                  | TPR (↑) | FPR (↓) | F1 (↑) | Latency OH (↓) |
|-------------------------|---------|---------|--------|----------------|
| No Defense (Baseline)   | 0%      | 0%      | N/A    | 0%             |
| Perplexity Filter       | 22%     | 3%      | 0.36   | 5%             |
| Output Checker          | 78%     | 12%     | 0.83   | 8%             |
| **NeuralPulse (Ours)**  | **89%** | **7%**  | **0.91** | **15%**      |

**Deliverables (Phase 3):**
- [ ] Fully functional `NeuralPulseMonitor` class
- [ ] Evaluation script with all metrics
- [ ] Comparison results vs. baselines (table + plots)
- [ ] Runtime overhead analysis

---

## Phase 4: Publication Assets

### 4.1 Required Figures (Publication-Ready)

**Figure 1: The Waffling Signature**
```
Title: "Temporal Signatures of SECA Attacks vs Normal Prompts"
Layout: 2x2 grid
  (a) Entropy H(t) over time - Attack traces (red) vs Normal (blue)
  (b) Attention Dispersion A(t) over time - Attack vs Normal
  (c) Phase space: H vs A scatter plot
  (d) Heatmap: Token-by-token signals for example attack trace
```

**Figure 2: STL Formula Evaluation**
```
Title: "STL Monitor Robustness Curve"
X-axis: STL threshold parameter sweep (θ_H)
Y-axis: F1 Score
Lines: Different formulas (φ_1, φ_2, φ_3)
Goal: Show φ_3 (combined) achieves best F1
```

**Figure 3: Defense Performance**
```
Title: "Detection Rate vs False Positive Rate (ROC-style)"
X-axis: False Positive Rate
Y-axis: True Positive Rate (Detection Rate)
Curves: NeuralPulse vs Baselines
Goal: Show NeuralPulse dominates (top-left corner)
```

**Figure 4: Ablation Study**
```
Title: "Contribution of Each Signal"
Bar chart:
  - Entropy Only (φ_1): F1 = 0.73
  - Attention Only (φ_2): F1 = 0.68
  - Combined (φ_3): F1 = 0.91
Goal: Show both signals are necessary
```

**Figure 5: Runtime Overhead**
```
Title: "Inference Time Breakdown"
Stacked bar chart:
  - Baseline generation
  - Signal extraction
  - STL evaluation
Show: Total overhead < 20%
```

### 4.2 Required Tables

**Table 1: Dataset Statistics**
| Split      | # Attack | # Normal | Subjects                     |
|------------|----------|----------|------------------------------|
| Train      | 600      | 600      | 16 MMLU subjects (stratified)|
| Validation | 200      | 200      | Same distribution            |
| Test       | 200      | 200      | Same distribution            |

**Table 2: Signal Statistics (Attack vs Normal)**
(From Phase 2.2 - statistical analysis)

**Table 3: Main Results**
(From Phase 3.3 - comparison with baselines)

**Table 4: Ablation Study**
| Component Removed       | TPR | FPR | F1   |
|-------------------------|-----|-----|------|
| Full System (φ_3)       | 89% | 7%  | 0.91 |
| w/o Entropy (A only)    | 71% | 9%  | 0.80 |
| w/o Attention (H only)  | 76% | 11% | 0.82 |
| w/o STL (threshold)     | 64% | 15% | 0.73 |

**Table 5: Cross-Model Generalization**
| Victim Model   | TPR | FPR | F1   |
|----------------|-----|-----|------|
| Llama-3-8B     | 89% | 7%  | 0.91 |
| Llama-2-13B    | 85% | 8%  | 0.88 |
| Qwen-2.5-7B    | 82% | 9%  | 0.86 |
| GPT-4o-Mini    | 78% | 12% | 0.83 |

### 4.3 Ablation Studies

**Ablation 1: Signal Contribution**
```python
# experiments/ablation_signals.py
"""
Test each signal independently:
  - Entropy Only: Use φ_1 (waffling detection)
  - Attention Only: Use φ_2 (context detachment)
  - Combined: Use φ_3 (full system)
"""
```

**Ablation 2: Threshold Sensitivity**
```python
# experiments/ablation_thresholds.py
"""
Vary θ_H, θ_A independently to show:
  - System is robust to threshold choice (F1 doesn't collapse)
  - Optimal values are data-driven (not arbitrary)
"""
```

**Ablation 3: STL vs Simple Threshold**
```python
# experiments/ablation_stl.py
"""
Compare:
  - NeuralPulse (STL temporal logic)
  - Simple rule: "if H > θ_H then block" (no temporal reasoning)

Show: Temporal logic (STL) is crucial for capturing attack dynamics.
"""
```

**Ablation 4: Victim Model Generalization**
```python
# experiments/ablation_models.py
"""
Train NeuralPulse on Llama-3-8B, test on:
  - Same model (in-distribution)
  - Different size: Llama-2-13B
  - Different family: Qwen-2.5-7B
  - Black-box: GPT-4o-Mini

Show: Defense transfers across models (not overfitted).
"""
```

**Deliverables (Phase 4):**
- [ ] 5 publication-ready figures (high-res, consistent style)
- [ ] 5 LaTeX tables with results
- [ ] 4 ablation study scripts + results
- [ ] `paper/` directory with LaTeX source + figures

---

## Implementation Timeline (12 Weeks)

| Week | Phase             | Milestones                                              |
|------|-------------------|---------------------------------------------------------|
| 1    | Phase 1.1-1.2     | Environment setup, dataset prep                         |
| 2-3  | Phase 1.3         | LlamaHook implementation                                |
| 3-4  | Phase 1.4         | Trace generation (1000+ attack, 1000+ normal)           |
| 4-5  | Phase 2.1         | Visualization, signature discovery                      |
| 5-6  | Phase 2.2         | Statistical analysis                                    |
| 6-7  | Phase 2.3         | STL formula derivation + threshold tuning               |
| 7-8  | Phase 3.1         | NeuralPulse monitor implementation                      |
| 8-9  | Phase 3.2         | Evaluation metrics, test set results                    |
| 9-10 | Phase 3.3         | Baseline comparisons                                    |
| 10-11| Phase 4.1-4.2     | Generate all figures/tables                             |
| 11-12| Phase 4.3         | Ablation studies                                        |
| 12   | Finalization      | Paper writing, code cleanup, submission prep            |

---

## Directory Structure

```
Neural-Pulse/
├── .neuralpulse/
│   └── config.json                 # Project configuration (created)
├── infrastructure/
│   ├── k8s/
│   │   ├── neuralpulse-job.yaml    # Nautilus K8s job template
│   │   └── pvc-traces.yaml         # Persistent volume for traces
│   └── docker/
│       └── Dockerfile              # Custom container (if needed)
├── core/
│   ├── llama_hook.py               # Signal extraction hook
│   └── neuralpulse_monitor.py      # STL runtime monitor
├── datasets/
│   ├── seca_loader.py              # SECA attack dataset
│   └── mmlu_loader.py              # MMLU normal dataset
├── formulas/
│   └── stl_specifications.py       # STL formulas (φ_1, φ_2, φ_3)
├── analysis/
│   ├── visualize_traces.py         # Plot generation
│   └── statistical_tests.py        # Hypothesis testing
├── baselines/
│   ├── perplexity_filter.py        # Baseline 1
│   └── output_checker.py           # Baseline 2
├── evaluation/
│   └── metrics.py                  # TPR, FPR, F1 computation
├── experiments/
│   ├── 01_extract_traces.py        # Phase 1.4
│   ├── 02_tune_thresholds.py       # Phase 2.3
│   ├── 03_evaluate_defense.py      # Phase 3.2
│   ├── ablation_signals.py         # Phase 4.3
│   ├── ablation_thresholds.py
│   ├── ablation_stl.py
│   └── ablation_models.py
├── data/
│   ├── raw/                        # SECA + MMLU raw data
│   └── traces/                     # Generated signal traces
│       ├── attack/
│       └── normal/
├── results/
│   ├── figures/                    # Publication figures
│   ├── tables/                     # LaTeX tables
│   └── checkpoints/                # Saved models/formulas
├── paper/
│   ├── main.tex                    # LaTeX manuscript
│   ├── figures/                    # Symlink to results/figures
│   └── bibliography.bib
├── MASTER_PLAN.md                  # This document
└── README.md                       # Project overview
```

---

## Key Hypotheses to Validate

### Hypothesis 1: Temporal Signature Exists
**Claim:** SECA attacks exhibit a distinct temporal pattern (high entropy + low attention) detectable during generation.

**Validation:** Phase 2.2 statistical tests must show:
- p < 0.001 for difference in mean entropy (attack vs normal)
- p < 0.001 for difference in mean attention dispersion
- Cohen's d > 1.0 (large effect size)

### Hypothesis 2: STL Outperforms Static Checks
**Claim:** Temporal logic (STL) is superior to static threshold checks.

**Validation:** Ablation Study (Phase 4.3) must show:
- STL (φ_3): F1 > 0.85
- Simple threshold: F1 < 0.75
- Gain: Δ F1 > 0.10

### Hypothesis 3: Defense Generalizes Across Models
**Claim:** NeuralPulse trained on one model transfers to others.

**Validation:** Phase 4.3 (Table 5) must show:
- F1 drop < 15% when testing on different model families
- TPR > 75% on all tested models

---

## Success Criteria for Publication

### Minimum Viable Paper (MiniR)
- [ ] TPR > 80% on SECA attacks
- [ ] FPR < 15% on normal prompts
- [ ] 4 publication figures showing signature + defense performance
- [ ] 2 baseline comparisons
- [ ] 1 ablation study (signals)

### Strong Paper (Top-tier Venue)
- [ ] TPR > 85%, FPR < 10%
- [ ] 5 publication figures + 5 tables
- [ ] 3 baseline comparisons
- [ ] 4 comprehensive ablation studies
- [ ] Cross-model generalization results
- [ ] Runtime overhead < 20%
- [ ] Open-source code + reproduc reproduction instructions

---

## Contingency Plans

### If Hypothesis 1 Fails (No Clear Signature)
**Backup Plan:** Explore additional signals:
- Token probability variance
- Hidden state norm
- Layer-wise activation patterns

### If Performance is Below Target (F1 < 0.80)
**Backup Plan:**
- Ensemble multiple STL formulas (voting)
- Add learnable parameters to STL thresholds
- Use lightweight ML classifier on top of signals

### If Latency Overhead is Too High (> 30%)
**Backup Plan:**
- Optimize signal extraction (batch operations)
- Approximate attention computation (sample subset of heads)
- Use early stopping (detect violations faster)

---

## Next Steps (Immediate Actions)

1. **Set up Nautilus K8s job** (infrastructure/k8s/neuralpulse-job.yaml)
2. **Clone SECA repository** and verify dataset access
3. **Implement LlamaHook v0.1** (entropy extraction only)
4. **Run initial trace extraction** (10 attack + 10 normal samples)
5. **Verify signals are logged correctly** (manual inspection)

**First Concrete Milestone (Week 1):**
Generate and visualize 20 sample traces to confirm signal extraction works.

---

## References for Implementation

**STL Libraries:**
- rtamt: https://github.com/nickovic/rtamt
- Documentation: https://rtamt.readthedocs.io/

**SECA Attack:**
- Paper: https://arxiv.org/abs/2510.04398
- Code: https://github.com/Buyun-Liang/SECA

**MMLU Dataset:**
- HuggingFace: `cais/mmlu`

**Nautilus K8s:**
- Documentation: See `mizzou_a100_guide.md`

---

**End of Master Plan**
**Next Action:** Proceed to Phase 1.1 (Environment Setup)
