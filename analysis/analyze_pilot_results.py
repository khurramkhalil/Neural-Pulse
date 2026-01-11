#!/usr/bin/env python3
"""
Critical Analysis of Phase 1 Pilot Results (100 SECA Attacks)

This script performs comprehensive analysis of the pilot attack generation results,
including success rate analysis, score distributions, equivalence analysis, and
identification of potential issues.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class PilotResultsAnalyzer:
    """Analyze SECA attack generation pilot results"""

    def __init__(self, results_path: str):
        """Load and initialize analyzer"""
        with open(results_path) as f:
            self.data = json.load(f)

        self.attacks = self.data['attacks']
        self.stats = self.data['statistics']
        self.generator_config = self.data['generator']

        logger.info(f"Loaded {len(self.attacks)} attack results")

    def analyze_success_rate(self) -> Dict:
        """Analyze attack success rates"""
        total = len(self.attacks)
        successful = sum(1 for a in self.attacks if a['success'])
        failed = total - successful

        success_rate = successful / total if total > 0 else 0

        # Analyze why attacks failed
        failed_attacks = [a for a in self.attacks if not a['success']]

        # Check if failed attacks reached max iterations
        max_iter_failures = sum(1 for a in failed_attacks
                               if a['iterations'] >= self.generator_config['max_iterations'])

        # Analyze scores of failed attacks
        failed_adv_scores = [a['adversarial_score'] for a in failed_attacks]
        failed_equiv_scores = [a['equivalence_score'] for a in failed_attacks]

        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'max_iter_failures': max_iter_failures,
            'failed_adversarial_scores': {
                'mean': np.mean(failed_adv_scores) if failed_adv_scores else 0,
                'median': np.median(failed_adv_scores) if failed_adv_scores else 0,
                'min': np.min(failed_adv_scores) if failed_adv_scores else 0,
                'max': np.max(failed_adv_scores) if failed_adv_scores else 0
            },
            'failed_equivalence_scores': {
                'mean': np.mean(failed_equiv_scores) if failed_equiv_scores else 0,
                'median': np.median(failed_equiv_scores) if failed_equiv_scores else 0,
                'min': np.min(failed_equiv_scores) if failed_equiv_scores else 0,
                'max': np.max(failed_equiv_scores) if failed_equiv_scores else 0
            }
        }

    def analyze_score_distributions(self) -> Dict:
        """Analyze adversarial and equivalence score distributions"""
        successful_attacks = [a for a in self.attacks if a['success']]

        adv_scores = [a['adversarial_score'] for a in successful_attacks]
        equiv_scores = [a['equivalence_score'] for a in successful_attacks]

        return {
            'adversarial_scores': {
                'mean': np.mean(adv_scores),
                'median': np.median(adv_scores),
                'std': np.std(adv_scores),
                'min': np.min(adv_scores),
                'max': np.max(adv_scores),
                'percentile_25': np.percentile(adv_scores, 25),
                'percentile_75': np.percentile(adv_scores, 75),
                'values': adv_scores
            },
            'equivalence_scores': {
                'mean': np.mean(equiv_scores),
                'median': np.median(equiv_scores),
                'std': np.std(equiv_scores),
                'min': np.min(equiv_scores),
                'max': np.max(equiv_scores),
                'percentile_25': np.percentile(equiv_scores, 25),
                'percentile_75': np.percentile(equiv_scores, 75),
                'values': equiv_scores
            }
        }

    def analyze_prompt_modifications(self) -> Dict:
        """Analyze how prompts were modified"""
        successful_attacks = [a for a in self.attacks if a['success']]

        # Categorize modifications
        modifications = {
            'identical': [],  # Adversarial same as original
            'minor': [],      # Very similar (equiv > 0.98)
            'moderate': [],   # Moderate change (0.95 < equiv <= 0.98)
            'substantial': [] # Substantial change (equiv <= 0.95)
        }

        for attack in successful_attacks:
            orig = attack['original_prompt']
            adv = attack['adversarial_prompt']
            equiv = attack['equivalence_score']

            if orig == adv:
                modifications['identical'].append(attack)
            elif equiv > 0.98:
                modifications['minor'].append(attack)
            elif equiv > 0.95:
                modifications['moderate'].append(attack)
            else:
                modifications['substantial'].append(attack)

        return {
            'identical_count': len(modifications['identical']),
            'minor_count': len(modifications['minor']),
            'moderate_count': len(modifications['moderate']),
            'substantial_count': len(modifications['substantial']),
            'modifications': modifications
        }

    def identify_problematic_attacks(self, threshold: float = 0.01) -> List[Dict]:
        """
        Identify potentially problematic successful attacks

        Problematic = adversarial_score > threshold (not really semantic attack)
        """
        successful_attacks = [a for a in self.attacks if a['success']]

        problematic = [a for a in successful_attacks
                      if a['adversarial_score'] > threshold]

        # Sort by adversarial score (highest first)
        problematic.sort(key=lambda x: x['adversarial_score'], reverse=True)

        return problematic

    def analyze_iterations(self) -> Dict:
        """Analyze iteration patterns"""
        iterations = [a['iterations'] for a in self.attacks]

        successful_iters = [a['iterations'] for a in self.attacks if a['success']]
        failed_iters = [a['iterations'] for a in self.attacks if not a['success']]

        return {
            'overall': {
                'mean': np.mean(iterations),
                'median': np.median(iterations),
                'min': np.min(iterations),
                'max': np.max(iterations)
            },
            'successful': {
                'mean': np.mean(successful_iters) if successful_iters else 0,
                'median': np.median(successful_iters) if successful_iters else 0,
            },
            'failed': {
                'mean': np.mean(failed_iters) if failed_iters else 0,
                'median': np.median(failed_iters) if failed_iters else 0,
            }
        }

    def generate_visualizations(self, output_dir: str = 'results/pilot_analysis'):
        """Generate comprehensive visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        successful_attacks = [a for a in self.attacks if a['success']]
        failed_attacks = [a for a in self.attacks if not a['success']]

        # 1. Success Rate Pie Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#e74c3c']
        sizes = [len(successful_attacks), len(failed_attacks)]
        labels = [f'Successful\n({len(successful_attacks)}/100)',
                 f'Failed\n({len(failed_attacks)}/100)']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax.set_title('SECA Attack Success Rate (Pilot: 100 Attacks)',
                    fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/success_rate.png', bbox_inches='tight')
        plt.close()

        # 2. Adversarial Score Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Log scale histogram
        adv_scores = [a['adversarial_score'] for a in successful_attacks]
        axes[0].hist(np.log10(np.array(adv_scores) + 1e-10), bins=30,
                    color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('log10(Adversarial Score)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Adversarial Score Distribution (Log Scale)',
                         fontsize=12, weight='bold')
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(adv_scores, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#3498db', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Adversarial Score', fontsize=11)
        axes[1].set_title('Adversarial Score Box Plot', fontsize=12, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_xticks([])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/adversarial_scores.png', bbox_inches='tight')
        plt.close()

        # 3. Equivalence Score Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        equiv_scores = [a['equivalence_score'] for a in successful_attacks]
        ax.hist(equiv_scores, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.axvline(0.95, color='red', linestyle='--', linewidth=2,
                  label='Threshold (0.95)', alpha=0.7)
        ax.set_xlabel('Equivalence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Semantic Equivalence Score Distribution',
                    fontsize=13, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equivalence_scores.png', bbox_inches='tight')
        plt.close()

        # 4. Scatter: Adversarial vs Equivalence
        fig, ax = plt.subplots(figsize=(10, 8))

        success_adv = [a['adversarial_score'] for a in successful_attacks]
        success_equiv = [a['equivalence_score'] for a in successful_attacks]

        ax.scatter(success_equiv, success_adv, c='#2ecc71',
                  alpha=0.6, s=50, label='Successful', edgecolors='black', linewidth=0.5)

        if failed_attacks:
            failed_adv = [a['adversarial_score'] for a in failed_attacks]
            failed_equiv = [a['equivalence_score'] for a in failed_attacks]
            ax.scatter(failed_equiv, failed_adv, c='#e74c3c',
                      alpha=0.6, s=50, label='Failed', edgecolors='black', linewidth=0.5)

        ax.axvline(0.95, color='blue', linestyle='--', linewidth=1.5,
                  label='Equiv Threshold', alpha=0.5)
        ax.axhline(0.01, color='orange', linestyle='--', linewidth=1.5,
                  label='Adv Threshold', alpha=0.5)

        ax.set_xlabel('Equivalence Score', fontsize=12)
        ax.set_ylabel('Adversarial Score', fontsize=12)
        ax.set_title('Adversarial vs Equivalence Scores', fontsize=13, weight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_correlation.png', bbox_inches='tight')
        plt.close()

        # 5. Iteration Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        all_iters = [a['iterations'] for a in self.attacks]
        axes[0].hist(all_iters, bins=15, color='#e67e22', alpha=0.7, edgecolor='black')
        axes[0].axvline(30, color='red', linestyle='--', linewidth=2,
                       label='Max Iterations', alpha=0.7)
        axes[0].set_xlabel('Iterations', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Iteration Distribution', fontsize=12, weight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Comparison
        success_iters = [a['iterations'] for a in successful_attacks]
        failed_iters = [a['iterations'] for a in failed_attacks]

        axes[1].boxplot([success_iters, failed_iters],
                       labels=['Successful', 'Failed'],
                       patch_artist=True,
                       boxprops=dict(alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Iterations', fontsize=11)
        axes[1].set_title('Iterations: Success vs Failure', fontsize=12, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/iterations.png', bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}/")

    def generate_report(self, output_path: str = 'results/pilot_analysis/PILOT_ANALYSIS.md'):
        """Generate comprehensive markdown report"""
        success_analysis = self.analyze_success_rate()
        score_analysis = self.analyze_score_distributions()
        mod_analysis = self.analyze_prompt_modifications()
        iter_analysis = self.analyze_iterations()
        problematic = self.identify_problematic_attacks(threshold=0.01)

        report = f"""# Phase 1 Pilot Results: Critical Analysis

**Date**: 2026-01-10
**Dataset**: 100 SECA Attack Attempts
**Generator**: {self.generator_config['proposer_provider']} ({self.generator_config['proposer_model']})
**Checker**: {self.generator_config['checker']}

---

## Executive Summary

### Overall Performance
- **Total Attacks**: {success_analysis['total']}
- **Successful**: {success_analysis['successful']} ({success_analysis['success_rate']*100:.1f}%)
- **Failed**: {success_analysis['failed']} ({(1-success_analysis['success_rate'])*100:.1f}%)
- **Average Iterations**: {iter_analysis['overall']['mean']:.1f}

### Key Findings
"""

        # Add key findings
        high_adv_score_count = len([a for a in self.attacks if a['success'] and a['adversarial_score'] > 0.1])
        low_equiv_count = len([a for a in self.attacks if a['success'] and a['equivalence_score'] < 0.95])

        report += f"""
1. **Success Rate Issue**: Only {success_analysis['success_rate']*100:.1f}% success rate indicates difficulty in generating truly semantic attacks
2. **Problematic Attacks**: {len(problematic)} successful attacks have adversarial_score > 0.01 (should be near 0)
3. **High Adversarial Scores**: {high_adv_score_count} attacks have adversarial_score > 0.1 (likely NOT semantic attacks)
4. **Equivalence Issues**: {low_equiv_count} attacks have equivalence < 0.95 (may not be semantically equivalent)
5. **Iteration Exhaustion**: {success_analysis['max_iter_failures']} failures hit max iterations (30)

---

## 1. Success Rate Analysis

### Breakdown
- Successful attacks: {success_analysis['successful']}/{success_analysis['total']} ({success_analysis['success_rate']*100:.1f}%)
- Failed attacks: {success_analysis['failed']}/{success_analysis['total']} ({(1-success_analysis['success_rate'])*100:.1f}%)
- Max iteration failures: {success_analysis['max_iter_failures']}/{success_analysis['failed']} failed attempts

### Why Attacks Failed

Failed attacks analysis:
- **Mean adversarial score**: {success_analysis['failed_adversarial_scores']['mean']:.6f}
- **Median adversarial score**: {success_analysis['failed_adversarial_scores']['median']:.6f}
- **Mean equivalence score**: {success_analysis['failed_equivalence_scores']['mean']:.4f}

**Critical Observation**: Failed attacks often have LOW adversarial scores but still failed. This suggests:
- The checker model (DeBERTa-large-mnli) might be TOO STRICT
- Some prompts may be inherently difficult to attack semantically
- The equivalence checker might be rejecting valid semantic paraphrases

---

## 2. Score Distribution Analysis

### Adversarial Scores (Successful Attacks)
- **Mean**: {score_analysis['adversarial_scores']['mean']:.6f}
- **Median**: {score_analysis['adversarial_scores']['median']:.6f}
- **Std Dev**: {score_analysis['adversarial_scores']['std']:.6f}
- **Min**: {score_analysis['adversarial_scores']['min']:.6f}
- **Max**: {score_analysis['adversarial_scores']['max']:.6f}
- **Q1**: {score_analysis['adversarial_scores']['percentile_25']:.6f}
- **Q3**: {score_analysis['adversarial_scores']['percentile_75']:.6f}

**CRITICAL ISSUE**: Mean adversarial score of {score_analysis['adversarial_scores']['mean']:.6f} is HIGHER than ideal (should be < 0.001).

**Distribution Characteristics**:
- Max score of {score_analysis['adversarial_scores']['max']:.4f} indicates some "successful" attacks are NOT truly semantic
- {len([s for s in score_analysis['adversarial_scores']['values'] if s > 0.1])} attacks have score > 0.1 (likely failed semantic constraint)
- {len([s for s in score_analysis['adversarial_scores']['values'] if s > 0.05])} attacks have score > 0.05

### Equivalence Scores (Successful Attacks)
- **Mean**: {score_analysis['equivalence_scores']['mean']:.4f}
- **Median**: {score_analysis['equivalence_scores']['median']:.4f}
- **Std Dev**: {score_analysis['equivalence_scores']['std']:.4f}
- **Min**: {score_analysis['equivalence_scores']['min']:.4f}
- **Max**: {score_analysis['equivalence_scores']['max']:.4f}
- **Q1**: {score_analysis['equivalence_scores']['percentile_25']:.4f}
- **Q3**: {score_analysis['equivalence_scores']['percentile_75']:.4f}

**OBSERVATION**: High equivalence scores (mean {score_analysis['equivalence_scores']['mean']:.4f}) indicate good semantic preservation in successful attacks.

---

## 3. Prompt Modification Analysis

### Modification Categories
- **Identical** (prompt unchanged): {mod_analysis['identical_count']} ({mod_analysis['identical_count']/success_analysis['successful']*100:.1f}%)
- **Minor** (equiv > 0.98): {mod_analysis['minor_count']} ({mod_analysis['minor_count']/success_analysis['successful']*100:.1f}%)
- **Moderate** (0.95 < equiv ≤ 0.98): {mod_analysis['moderate_count']} ({mod_analysis['moderate_count']/success_analysis['successful']*100:.1f}%)
- **Substantial** (equiv ≤ 0.95): {mod_analysis['substantial_count']} ({mod_analysis['substantial_count']/success_analysis['successful']*100:.1f}%)

**MAJOR CONCERN**: {mod_analysis['identical_count']} attacks where adversarial == original prompt!

These are NOT attacks at all - they indicate the generator failed to modify the prompt but still marked as "successful".

---

## 4. Problematic Attacks (adversarial_score > 0.01)

Found **{len(problematic)}** problematic successful attacks (should have score near 0 for semantic attacks).

### Top 10 Most Problematic:
"""

        for i, attack in enumerate(problematic[:10], 1):
            report += f"""
#### {i}. Adversarial Score: {attack['adversarial_score']:.6f} | Equivalence: {attack['equivalence_score']:.4f}

**Original**: {attack['original_prompt'][:100]}...

**Adversarial**: {attack['adversarial_prompt'][:100]}...

**Issue**: Adversarial score {attack['adversarial_score']:.6f} is TOO HIGH for a semantic attack.
"""

        report += f"""

---

## 5. Iteration Analysis

### Overall
- Mean iterations: {iter_analysis['overall']['mean']:.1f}
- Median iterations: {iter_analysis['overall']['median']:.1f}
- Min iterations: {iter_analysis['overall']['min']}
- Max iterations: {iter_analysis['overall']['max']}

### Successful vs Failed
- **Successful attacks**: Mean = {iter_analysis['successful']['mean']:.1f}, Median = {iter_analysis['successful']['median']:.1f}
- **Failed attacks**: Mean = {iter_analysis['failed']['mean']:.1f}, Median = {iter_analysis['failed']['median']:.1f}

**OBSERVATION**: All attacks ran for {iter_analysis['overall']['mean']:.0f} iterations (max allowed). This suggests:
- Max iterations might be set as a hard limit rather than stopping when attack succeeds
- OR: Most attacks need all 30 iterations to succeed/fail

---

## 6. Critical Issues Identified

### Issue 1: High Adversarial Scores
**Severity**: CRITICAL

{len(problematic)} "successful" attacks have adversarial_score > 0.01, with {len([a for a in problematic if a['adversarial_score'] > 0.1])} having scores > 0.1.

**Impact**: These are NOT truly semantic attacks - they modify the prompt in ways that ARE detectable by the checker.

**Recommendation**:
- Tighten adversarial score threshold (e.g., < 0.001)
- Investigate why checker is accepting high-score attacks

### Issue 2: Identical Prompts
**Severity**: CRITICAL

{mod_analysis['identical_count']} attacks have identical original and adversarial prompts.

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

{success_analysis['max_iter_failures']} failures hit maximum iterations.

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

**Valid Attacks**: {len([a for a in self.attacks if a['success'] and a['adversarial_score'] < 0.01 and a['original_prompt'] != a['adversarial_prompt']])} / {success_analysis['successful']}

**Effective Success Rate**: {len([a for a in self.attacks if a['success'] and a['adversarial_score'] < 0.01 and a['original_prompt'] != a['adversarial_prompt']]) / success_analysis['total'] * 100:.1f}%

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
"""

        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {output_path}")

        return report


def main():
    """Run comprehensive analysis"""
    analyzer = PilotResultsAnalyzer('seca_attacks_pilot_100.json')

    print("\n" + "="*80)
    print("PHASE 1 PILOT RESULTS: CRITICAL ANALYSIS")
    print("="*80 + "\n")

    # 1. Success Rate Analysis
    print("\n1. SUCCESS RATE ANALYSIS")
    print("-" * 40)
    success_analysis = analyzer.analyze_success_rate()
    print(f"Total attacks: {success_analysis['total']}")
    print(f"Successful: {success_analysis['successful']} ({success_analysis['success_rate']*100:.1f}%)")
    print(f"Failed: {success_analysis['failed']} ({(1-success_analysis['success_rate'])*100:.1f}%)")
    print(f"Max iteration failures: {success_analysis['max_iter_failures']}")

    # 2. Score Analysis
    print("\n2. SCORE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    score_analysis = analyzer.analyze_score_distributions()
    print(f"\nAdversarial Scores (successful attacks):")
    print(f"  Mean: {score_analysis['adversarial_scores']['mean']:.6f}")
    print(f"  Median: {score_analysis['adversarial_scores']['median']:.6f}")
    print(f"  Max: {score_analysis['adversarial_scores']['max']:.6f}")

    print(f"\nEquivalence Scores (successful attacks):")
    print(f"  Mean: {score_analysis['equivalence_scores']['mean']:.4f}")
    print(f"  Median: {score_analysis['equivalence_scores']['median']:.4f}")
    print(f"  Min: {score_analysis['equivalence_scores']['min']:.4f}")

    # 3. Modification Analysis
    print("\n3. PROMPT MODIFICATION ANALYSIS")
    print("-" * 40)
    mod_analysis = analyzer.analyze_prompt_modifications()
    print(f"Identical prompts: {mod_analysis['identical_count']}")
    print(f"Minor modifications: {mod_analysis['minor_count']}")
    print(f"Moderate modifications: {mod_analysis['moderate_count']}")
    print(f"Substantial modifications: {mod_analysis['substantial_count']}")

    # 4. Problematic Attacks
    print("\n4. PROBLEMATIC ATTACKS (adversarial_score > 0.01)")
    print("-" * 40)
    problematic = analyzer.identify_problematic_attacks(threshold=0.01)
    print(f"Found {len(problematic)} problematic attacks")
    if problematic:
        print(f"\nTop 3 worst:")
        for i, attack in enumerate(problematic[:3], 1):
            print(f"  {i}. Score: {attack['adversarial_score']:.6f}, Equiv: {attack['equivalence_score']:.4f}")

    # 5. Generate Visualizations
    print("\n5. GENERATING VISUALIZATIONS")
    print("-" * 40)
    analyzer.generate_visualizations()

    # 6. Generate Report
    print("\n6. GENERATING COMPREHENSIVE REPORT")
    print("-" * 40)
    analyzer.generate_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  - Visualizations: results/pilot_analysis/*.png")
    print("  - Report: results/pilot_analysis/PILOT_ANALYSIS.md")
    print("\n")


if __name__ == '__main__':
    main()
