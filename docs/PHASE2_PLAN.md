# Phase 2: Diagnosis & Formula Mining - Implementation Plan

## Overview

**Goal**: Analyze signal patterns from validated traces to derive optimal STL formula parameters.

**Duration**: Weeks 4-6 (3 weeks)

**Input**: 2000 validated traces with entropy/attention signals (Phase 1 output)

**Output**:
- Signal visualizations showing attack vs normal patterns
- Statistical analysis of signal distributions
- Optimized STL formula thresholds (θ_H, θ_A, T)
- Three STL formulas: φ₁ (waffling), φ₂ (detachment), φ₃ (combined)

---

## Phase 2 Components

### 1. Signal Visualization (`analysis/visualize_signals.py`)

**Purpose**: Visual analysis of entropy and attention patterns

**Features**:
- Plot individual traces (entropy + attention over time)
- Compare attack vs normal distributions
- Heatmaps showing temporal patterns
- Statistical overlays (mean, std, percentiles)

**Tests** (`tests/unit/test_visualize_signals.py`):
- Plot generation doesn't crash
- Correct data used for each plot type
- Figure dimensions and labels are correct
- Edge cases (empty data, single trace)

---

### 2. Statistical Analysis (`analysis/statistical_analysis.py`)

**Purpose**: Compute signal statistics to identify discriminative features

**Features**:
- Distribution analysis (mean, median, std, percentiles)
- Temporal features (max, min, variance, duration)
- Comparative statistics (attack vs normal)
- ROC analysis for threshold selection

**Tests** (`tests/unit/test_statistical_analysis.py`):
- Statistical calculations are mathematically correct
- Distribution comparisons are accurate
- Edge cases (identical distributions, empty data)
- ROC curve computation validates

---

### 3. STL Formula Mining (`analysis/formula_mining.py`)

**Purpose**: Derive optimal STL formula parameters through grid search

**Features**:
- Grid search over threshold space (θ_H, θ_A, T)
- Evaluate each parameter combination on validation set
- Optimize for F1-score (balance TPR and FPR)
- Output best parameters for each formula

**Tests** (`tests/unit/test_formula_mining.py`):
- Grid search explores full parameter space
- Optimization finds correct maxima
- Edge cases (all correct, all wrong, random)

---

### 4. STL Formula Implementation (`core/stl_formulas.py`)

**Purpose**: Implement three STL formulas using rtamt library

**Formulas**:
```
φ₁: Eventually[0,T](Always[t,t+3](H(t) > θ_H))
    # Waffling: Sustained high entropy

φ₂: Eventually[0,T](Always[t,t+3](A(t) < θ_A))
    # Detachment: Sustained low context attention

φ₃: Eventually[0,T](Always[t,t+3](H(t) > θ_H AND A(t) < θ_A))
    # Combined: Both waffling AND detachment
```

**Tests** (`tests/unit/test_stl_formulas.py`):
- Formula syntax is valid (rtamt parses correctly)
- Evaluation matches expected behavior
- Edge cases (empty signals, constant signals)
- Mathematical properties verified

---

## Implementation Order

### Week 4: Visualization & Statistics
1. **Day 1-2**: Implement `visualize_signals.py` + tests
2. **Day 3-4**: Implement `statistical_analysis.py` + tests
3. **Day 5**: Integration testing, bug fixes

### Week 5: Formula Mining
1. **Day 1-2**: Implement `formula_mining.py` + tests
2. **Day 3-4**: Implement `stl_formulas.py` + tests
3. **Day 5**: Optimize grid search, validate results

### Week 6: Validation & Documentation
1. **Day 1-2**: Run full pipeline on 2000 traces
2. **Day 3**: Generate final visualizations and statistics
3. **Day 4**: Document findings, create Phase 2 report
4. **Day 5**: Prepare for Phase 3 (Monitor implementation)

---

## Expected Outputs

### Visualizations
- `paper/figures/entropy_distribution_attack_vs_normal.png`
- `paper/figures/attention_distribution_attack_vs_normal.png`
- `paper/figures/temporal_pattern_heatmap.png`
- `paper/figures/sample_traces_attack.png`
- `paper/figures/sample_traces_normal.png`

### Statistics
- `analysis/results/signal_statistics.json`
- `analysis/results/threshold_roc_curves.png`
- `analysis/results/discriminative_features.csv`

### Formula Parameters
- `analysis/results/optimal_parameters.json`:
```json
{
  "formula_1_waffling": {
    "theta_H": 2.3,
    "T": 50,
    "validation_f1": 0.87
  },
  "formula_2_detachment": {
    "theta_A": 0.35,
    "T": 50,
    "validation_f1": 0.84
  },
  "formula_3_combined": {
    "theta_H": 2.1,
    "theta_A": 0.38,
    "T": 50,
    "validation_f1": 0.91
  }
}
```

---

## Success Criteria

- [ ] All visualization tests pass (>95% coverage)
- [ ] All statistical tests pass (>95% coverage)
- [ ] All formula mining tests pass (>95% coverage)
- [ ] All STL formula tests pass (>95% coverage)
- [ ] Optimal parameters found with F1 > 0.85
- [ ] Visualizations clearly show attack vs normal patterns
- [ ] No bugs or crashes in full pipeline run

---

## Key Technical Decisions

### 1. Visualization Library
**Choice**: matplotlib + seaborn
- Reason: Publication-quality figures, extensive customization
- Alternative: plotly (interactive, but not needed for paper)

### 2. Statistical Methods
**Choice**: scipy.stats for distributions, sklearn for ROC
- Reason: Well-tested, standard in ML research
- Edge cases: Use robust statistics (median, IQR) for outliers

### 3. Grid Search Strategy
**Choice**: Exhaustive grid search with cross-validation
- Parameter ranges:
  - θ_H: [1.5, 3.5] in steps of 0.1 (20 values)
  - θ_A: [0.2, 0.6] in steps of 0.05 (8 values)
  - T: [30, 80] in steps of 10 (6 values)
- Total: 20 × 8 × 6 = 960 combinations per formula
- Optimization metric: F1-score (harmonic mean of precision/recall)

### 4. STL Library
**Choice**: rtamt (Runtime Monitoring of STL)
- Reason: Python bindings, efficient evaluation, well-documented
- Alternative: stlpy (less mature, fewer features)

---

## Testing Strategy

### Unit Tests (Per Module)
- Mathematical correctness (formulas, statistics)
- Edge cases (empty, single value, constant)
- Error handling (invalid input, missing data)
- Output format validation (correct types, ranges)

### Integration Tests
- End-to-end pipeline: traces → visualization → statistics → formulas
- Cross-module compatibility
- Data format consistency
- Performance benchmarks (< 5min for 2000 traces)

### Validation Tests
- Sanity checks (attack traces have higher entropy than normal)
- Threshold verification (selected thresholds achieve target TPR/FPR)
- Visual inspection (plots show expected patterns)

---

## Next Steps (After Phase 2)

**Phase 3**: STL Monitor & Defense
- Implement `NeuralPulseMonitor` class
- Real-time signal evaluation during generation
- Blocking mechanism when formula violated
- Baseline comparisons (Semantic Entropy, SelfCheckGPT)

Let's begin implementation! 🚀
