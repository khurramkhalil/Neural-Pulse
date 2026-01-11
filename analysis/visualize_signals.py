"""
Signal Visualization Module for Neural Pulse

Visualizes entropy and attention signals from validated traces to identify
patterns that distinguish SECA attacks from normal prompts.

Key visualizations:
1. Distribution plots (attack vs normal)
2. Temporal traces (sample trajectories)
3. Heatmaps (signal patterns over time)
4. Statistical overlays (mean, std, percentiles)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
sns.set_palette("husl")


class SignalVisualizer:
    """
    Visualizes entropy and attention signals from validated traces.

    Supports multiple plot types:
    - Distribution comparisons (attack vs normal)
    - Individual trace plots
    - Temporal pattern heatmaps
    - Statistical summaries
    """

    def __init__(self, output_dir: str = "paper/figures"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SignalVisualizer initialized. Figures will be saved to: {self.output_dir}")

    def load_validated_traces(self, traces_path: str, validation_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load traces and split by hallucination status.

        Args:
            traces_path: Path to traces JSON (from LlamaHook)
            validation_path: Path to validation results (from OracleValidator)

        Returns:
            Tuple of (hallucination_traces, normal_traces)
        """
        # Load traces
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
            # Handle both formats: {'traces': [...]} or just [...]
            if isinstance(traces_data, dict):
                traces = traces_data.get('traces', traces_data)
            else:
                traces = traces_data

        # Load validation results
        with open(validation_path, 'r') as f:
            validation_data = json.load(f)
            # Handle both formats: {'validation_results': [...]} or just [...]
            if isinstance(validation_data, dict):
                validation_results = validation_data.get('validation_results', validation_data)
            else:
                validation_results = validation_data

        assert len(traces) == len(validation_results), \
            f"Trace count ({len(traces)}) != validation count ({len(validation_results)})"

        # Split by hallucination status
        hallucination_traces = []
        normal_traces = []

        for trace, validation in zip(traces, validation_results):
            if validation['is_hallucination']:
                hallucination_traces.append(trace)
            else:
                normal_traces.append(trace)

        logger.info(f"Loaded {len(hallucination_traces)} hallucination traces, "
                   f"{len(normal_traces)} normal traces")

        return hallucination_traces, normal_traces

    def plot_distribution_comparison(
        self,
        hallucination_traces: List[Dict],
        normal_traces: List[Dict],
        signal_name: str = 'entropy',
        metric: str = 'mean',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution comparison between attack and normal traces.

        Args:
            hallucination_traces: List of hallucination trace dicts
            normal_traces: List of normal trace dicts
            signal_name: 'entropy' or 'attention'
            metric: 'mean', 'max', 'min', or 'std'
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        signal_key = f'{signal_name}_trace'

        # Extract metrics for each group
        hallucination_values = []
        normal_values = []

        for trace in hallucination_traces:
            signal = trace.get(signal_key, [])
            if len(signal) > 0:
                if metric == 'mean':
                    hallucination_values.append(np.mean(signal))
                elif metric == 'max':
                    hallucination_values.append(np.max(signal))
                elif metric == 'min':
                    hallucination_values.append(np.min(signal))
                elif metric == 'std':
                    hallucination_values.append(np.std(signal))

        for trace in normal_traces:
            signal = trace.get(signal_key, [])
            if len(signal) > 0:
                if metric == 'mean':
                    normal_values.append(np.mean(signal))
                elif metric == 'max':
                    normal_values.append(np.max(signal))
                elif metric == 'min':
                    normal_values.append(np.min(signal))
                elif metric == 'std':
                    normal_values.append(np.std(signal))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms
        bins = 30
        alpha = 0.6

        ax.hist(hallucination_values, bins=bins, alpha=alpha,
               label=f'Attack (n={len(hallucination_values)})', color='red', edgecolor='black')
        ax.hist(normal_values, bins=bins, alpha=alpha,
               label=f'Normal (n={len(normal_values)})', color='blue', edgecolor='black')

        # Add vertical lines for means
        ax.axvline(np.mean(hallucination_values), color='darkred',
                  linestyle='--', linewidth=2, label=f'Attack mean: {np.mean(hallucination_values):.2f}')
        ax.axvline(np.mean(normal_values), color='darkblue',
                  linestyle='--', linewidth=2, label=f'Normal mean: {np.mean(normal_values):.2f}')

        # Labels and legend
        signal_label = signal_name.capitalize()
        metric_label = metric.capitalize()
        ax.set_xlabel(f'{metric_label} {signal_label}', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{signal_label} Distribution Comparison ({metric_label})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved distribution plot to: {save_path}")

        return fig

    def plot_sample_traces(
        self,
        traces: List[Dict],
        num_samples: int = 5,
        trace_type: str = 'attack',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sample individual traces showing entropy and attention over time.

        Args:
            traces: List of trace dicts
            num_samples: Number of sample traces to plot
            trace_type: 'attack' or 'normal' (for title)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        # Sample traces randomly
        if len(traces) > num_samples:
            sample_indices = np.random.choice(len(traces), num_samples, replace=False)
            sampled_traces = [traces[i] for i in sample_indices]
        else:
            sampled_traces = traces

        # Create subplots (4 columns: entropy, attention, perplexity, attention_entropy)
        # Use actual number of sampled traces, not requested num_samples
        actual_samples = len(sampled_traces)
        fig, axes = plt.subplots(actual_samples, 4, figsize=(20, 3*actual_samples))

        # Handle single sample case
        if actual_samples == 1:
            axes = axes.reshape(1, -1)

        for i, trace in enumerate(sampled_traces):
            entropy = trace.get('entropy_trace', [])
            attention = trace.get('attention_trace', [])
            perplexity = trace.get('perplexity_trace', [])
            attention_entropy = trace.get('attention_entropy_trace', [])

            # Plot entropy
            axes[i, 0].plot(entropy, linewidth=2, color='orange')
            axes[i, 0].set_ylabel('Entropy H(t)', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_title(f'Sample {i+1}: Entropy' if i == 0 else f'Sample {i+1}', fontsize=10)

            # Plot attention
            axes[i, 1].plot(attention, linewidth=2, color='purple')
            axes[i, 1].set_ylabel('Attention A(t)', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_title(f'Sample {i+1}: Attention' if i == 0 else f'Sample {i+1}', fontsize=10)

            # Plot perplexity
            if len(perplexity) > 0:
                axes[i, 2].plot(perplexity, linewidth=2, color='red')
                axes[i, 2].set_ylabel('Perplexity P(t)', fontsize=10)
                axes[i, 2].grid(True, alpha=0.3)
                axes[i, 2].set_title(f'Sample {i+1}: Perplexity' if i == 0 else f'Sample {i+1}', fontsize=10)

            # Plot attention entropy
            if len(attention_entropy) > 0:
                axes[i, 3].plot(attention_entropy, linewidth=2, color='green')
                axes[i, 3].set_ylabel('Attn Entropy H_a(t)', fontsize=10)
                axes[i, 3].grid(True, alpha=0.3)
                axes[i, 3].set_title(f'Sample {i+1}: Attn Entropy' if i == 0 else f'Sample {i+1}', fontsize=10)

            # Only show x-label on bottom row
            if i == num_samples - 1:
                axes[i, 0].set_xlabel('Token Position', fontsize=10)
                axes[i, 1].set_xlabel('Token Position', fontsize=10)
                axes[i, 2].set_xlabel('Token Position', fontsize=10)
                axes[i, 3].set_xlabel('Token Position', fontsize=10)

        fig.suptitle(f'{trace_type.capitalize()} Traces: Temporal Patterns',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved sample traces plot to: {save_path}")

        return fig

    def plot_temporal_heatmap(
        self,
        traces: List[Dict],
        signal_name: str = 'entropy',
        max_length: int = 100,
        trace_type: str = 'attack',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap showing temporal signal patterns across multiple traces.

        Args:
            traces: List of trace dicts
            signal_name: 'entropy' or 'attention'
            max_length: Maximum sequence length to plot
            trace_type: 'attack' or 'normal' (for title)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        signal_key = f'{signal_name}_trace'

        # Create matrix (traces × time)
        matrix = []
        for trace in traces:
            signal = trace.get(signal_key, [])
            if len(signal) > 0:
                # Pad or truncate to max_length
                if len(signal) < max_length:
                    padded = signal + [np.nan] * (max_length - len(signal))
                else:
                    padded = signal[:max_length]
                matrix.append(padded)

        matrix = np.array(matrix)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(matrix, cmap='RdYlBu_r', cbar_kws={'label': signal_name.capitalize()},
                   xticklabels=10, yticklabels=False, ax=ax)

        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Trace Index', fontsize=12)
        ax.set_title(f'{trace_type.capitalize()} Traces: {signal_name.capitalize()} Heatmap',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved heatmap to: {save_path}")

        return fig

    def plot_statistical_summary(
        self,
        hallucination_traces: List[Dict],
        normal_traces: List[Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot statistical summary comparing attack vs normal across both signals.

        Shows mean ± std for entropy and attention.

        Args:
            hallucination_traces: List of hallucination trace dicts
            normal_traces: List of normal trace dicts
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        # Compute statistics
        stats = {
            'entropy': {'attack': [], 'normal': []},
            'attention': {'attack': [], 'normal': []}
        }

        for trace in hallucination_traces:
            if len(trace.get('entropy_trace', [])) > 0:
                stats['entropy']['attack'].append(np.mean(trace['entropy_trace']))
            if len(trace.get('attention_trace', [])) > 0:
                stats['attention']['attack'].append(np.mean(trace['attention_trace']))

        for trace in normal_traces:
            if len(trace.get('entropy_trace', [])) > 0:
                stats['entropy']['normal'].append(np.mean(trace['entropy_trace']))
            if len(trace.get('attention_trace', [])) > 0:
                stats['attention']['normal'].append(np.mean(trace['attention_trace']))

        # Create bar plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        signals = ['entropy', 'attention']
        titles = ['Entropy H(t)', 'Attention A(t)']
        colors = ['orange', 'purple']

        for i, (signal, title, color) in enumerate(zip(signals, titles, colors)):
            attack_values = stats[signal]['attack']
            normal_values = stats[signal]['normal']

            x = np.arange(2)
            means = [np.mean(attack_values), np.mean(normal_values)]
            stds = [np.std(attack_values), np.std(normal_values)]

            axes[i].bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                       color=[color, 'lightblue'], edgecolor='black', linewidth=1.5)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(['Attack', 'Normal'], fontsize=11)
            axes[i].set_ylabel(f'Mean {title}', fontsize=11)
            axes[i].set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for j, (mean, std) in enumerate(zip(means, stds)):
                axes[i].text(j, mean + std + 0.05, f'{mean:.2f}±{std:.2f}',
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved statistical summary to: {save_path}")

        return fig

    def generate_all_visualizations(
        self,
        traces_path: str,
        validation_path: str,
        prefix: str = ""
    ):
        """
        Generate all standard visualizations for a dataset.

        Args:
            traces_path: Path to traces JSON
            validation_path: Path to validation results JSON
            prefix: Optional prefix for output filenames
        """
        logger.info("Generating all visualizations...")

        # Load data
        hallucination_traces, normal_traces = self.load_validated_traces(
            traces_path, validation_path
        )

        # 1. Distribution comparisons
        # Include all 4 signals
        for signal in ['entropy', 'attention', 'perplexity', 'attention_entropy']:
            for metric in ['mean', 'max']:
                filename = f'{prefix}{signal}_distribution_{metric}.png'
                save_path = self.output_dir / filename
                self.plot_distribution_comparison(
                    hallucination_traces, normal_traces,
                    signal_name=signal, metric=metric, save_path=save_path
                )

        # 2. Sample traces
        for trace_type, traces in [('attack', hallucination_traces), ('normal', normal_traces)]:
            filename = f'{prefix}sample_traces_{trace_type}.png'
            save_path = self.output_dir / filename
            self.plot_sample_traces(traces, num_samples=5, trace_type=trace_type, save_path=save_path)

        # 3. Heatmaps
        for signal in ['entropy', 'attention']:
            for trace_type, traces in [('attack', hallucination_traces), ('normal', normal_traces)]:
                filename = f'{prefix}heatmap_{signal}_{trace_type}.png'
                save_path = self.output_dir / filename
                self.plot_temporal_heatmap(traces, signal_name=signal,
                                          trace_type=trace_type, save_path=save_path)

        # 4. Statistical summary
        filename = f'{prefix}statistical_summary.png'
        save_path = self.output_dir / filename
        self.plot_statistical_summary(hallucination_traces, normal_traces, save_path=save_path)

        logger.info(f"All visualizations generated! Saved to: {self.output_dir}")


def main():
    """CLI interface for signal visualization"""
    import argparse

    parser = argparse.ArgumentParser(description='Neural Pulse Signal Visualization')
    parser.add_argument('--traces', type=str, required=True, help='Path to traces JSON')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation JSON')
    parser.add_argument('--output_dir', type=str, default='paper/figures',
                       help='Output directory for figures')
    parser.add_argument('--prefix', type=str, default='', help='Filename prefix')

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = SignalVisualizer(output_dir=args.output_dir)

    # Generate all visualizations
    visualizer.generate_all_visualizations(
        traces_path=args.traces,
        validation_path=args.validation,
        prefix=args.prefix
    )

    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()
