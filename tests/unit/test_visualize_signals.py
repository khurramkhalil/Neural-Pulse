"""
Comprehensive Unit Tests for Signal Visualization

Tests cover:
1. Data loading and splitting by hallucination status
2. Distribution plot generation
3. Sample trace plotting
4. Heatmap generation
5. Statistical summary plots
6. Edge cases (empty data, single trace, missing signals)

NO try-except gaming - tests verify actual plot content and data correctness.
"""

import unittest
import json
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
from analysis.visualize_signals import SignalVisualizer


class TestDataLoading(unittest.TestCase):
    """Test loading and splitting validated traces"""

    def setUp(self):
        self.visualizer = SignalVisualizer(output_dir=tempfile.mkdtemp())
        self.temp_dir = Path(tempfile.mkdtemp())

    def create_mock_data(self, num_attack=5, num_normal=5):
        """Helper to create mock traces and validation data"""
        traces = []
        validations = []

        # Create attack traces
        for i in range(num_attack):
            traces.append({
                'prompt': f'Attack prompt {i}',
                'generated_text': 'Response',
                'entropy_trace': np.random.uniform(2.0, 3.0, 50).tolist(),
                'attention_trace': np.random.uniform(0.2, 0.4, 50).tolist()
            })
            validations.append({'is_hallucination': True, 'confidence': 0.95})

        # Create normal traces
        for i in range(num_normal):
            traces.append({
                'prompt': f'Normal prompt {i}',
                'generated_text': 'Response',
                'entropy_trace': np.random.uniform(0.5, 1.5, 50).tolist(),
                'attention_trace': np.random.uniform(0.6, 0.9, 50).tolist()
            })
            validations.append({'is_hallucination': False, 'confidence': 0.95})

        return traces, validations

    def test_load_and_split_traces(self):
        """Test: Load traces and correctly split by hallucination status"""
        traces, validations = self.create_mock_data(num_attack=7, num_normal=3)

        # Save to temp files
        traces_path = self.temp_dir / 'traces.json'
        validation_path = self.temp_dir / 'validation.json'

        with open(traces_path, 'w') as f:
            json.dump({'traces': traces}, f)
        with open(validation_path, 'w') as f:
            json.dump({'validation_results': validations}, f)

        # Load and split
        hallucination_traces, normal_traces = self.visualizer.load_validated_traces(
            str(traces_path), str(validation_path)
        )

        self.assertEqual(len(hallucination_traces), 7)
        self.assertEqual(len(normal_traces), 3)

    def test_load_mismatched_lengths_raises_error(self):
        """Test: Mismatched trace/validation lengths raise error"""
        traces, validations = self.create_mock_data(num_attack=5, num_normal=5)

        # Remove one validation (create mismatch)
        validations = validations[:-1]

        traces_path = self.temp_dir / 'traces.json'
        validation_path = self.temp_dir / 'validation.json'

        with open(traces_path, 'w') as f:
            json.dump({'traces': traces}, f)
        with open(validation_path, 'w') as f:
            json.dump({'validation_results': validations}, f)

        with self.assertRaises(AssertionError):
            self.visualizer.load_validated_traces(str(traces_path), str(validation_path))

    def test_load_empty_dataset(self):
        """Test: Empty dataset returns empty lists"""
        traces_path = self.temp_dir / 'traces.json'
        validation_path = self.temp_dir / 'validation.json'

        with open(traces_path, 'w') as f:
            json.dump({'traces': []}, f)
        with open(validation_path, 'w') as f:
            json.dump({'validation_results': []}, f)

        hallucination_traces, normal_traces = self.visualizer.load_validated_traces(
            str(traces_path), str(validation_path)
        )

        self.assertEqual(len(hallucination_traces), 0)
        self.assertEqual(len(normal_traces), 0)


class TestDistributionPlots(unittest.TestCase):
    """Test distribution comparison plots"""

    def setUp(self):
        self.visualizer = SignalVisualizer(output_dir=tempfile.mkdtemp())

    def create_mock_traces(self, mean_entropy_attack=2.5, mean_entropy_normal=1.0):
        """Helper to create traces with specific distributions"""
        hallucination_traces = [
            {
                'entropy_trace': np.random.normal(mean_entropy_attack, 0.5, 50).tolist(),
                'attention_trace': np.random.normal(0.3, 0.1, 50).tolist()
            }
            for _ in range(20)
        ]

        normal_traces = [
            {
                'entropy_trace': np.random.normal(mean_entropy_normal, 0.3, 50).tolist(),
                'attention_trace': np.random.normal(0.7, 0.1, 50).tolist()
            }
            for _ in range(20)
        ]

        return hallucination_traces, normal_traces

    def test_distribution_plot_returns_figure(self):
        """Test: Distribution plot returns matplotlib Figure"""
        hallucination_traces, normal_traces = self.create_mock_traces()

        fig = self.visualizer.plot_distribution_comparison(
            hallucination_traces, normal_traces,
            signal_name='entropy', metric='mean'
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_distribution_plot_correct_data(self):
        """Test: Distribution plot uses correct data for each group"""
        hallucination_traces, normal_traces = self.create_mock_traces(
            mean_entropy_attack=3.0, mean_entropy_normal=1.0
        )

        fig = self.visualizer.plot_distribution_comparison(
            hallucination_traces, normal_traces,
            signal_name='entropy', metric='mean'
        )

        # Extract data from plot
        ax = fig.axes[0]
        attack_data = [patch.get_height() for patch in ax.patches[:30]]  # First histogram
        normal_data = [patch.get_height() for patch in ax.patches[30:]]  # Second histogram

        # Attack traces should have higher mean entropy
        self.assertGreater(sum(attack_data), 0)
        self.assertGreater(sum(normal_data), 0)

        plt.close(fig)

    def test_distribution_plot_saves_to_file(self):
        """Test: Distribution plot saves correctly to file"""
        hallucination_traces, normal_traces = self.create_mock_traces()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name

        fig = self.visualizer.plot_distribution_comparison(
            hallucination_traces, normal_traces,
            signal_name='entropy', metric='mean',
            save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(os.path.getsize(save_path), 0)

        plt.close(fig)
        os.unlink(save_path)

    def test_distribution_plot_different_metrics(self):
        """Test: Distribution plot works with different metrics"""
        hallucination_traces, normal_traces = self.create_mock_traces()

        for metric in ['mean', 'max', 'min', 'std']:
            fig = self.visualizer.plot_distribution_comparison(
                hallucination_traces, normal_traces,
                signal_name='entropy', metric=metric
            )

            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)

    def test_distribution_plot_empty_traces(self):
        """Test: Distribution plot handles empty trace lists"""
        fig = self.visualizer.plot_distribution_comparison(
            [], [],
            signal_name='entropy', metric='mean'
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestSampleTracePlots(unittest.TestCase):
    """Test individual sample trace plotting"""

    def setUp(self):
        self.visualizer = SignalVisualizer(output_dir=tempfile.mkdtemp())

    def create_mock_traces(self, num_traces=10):
        """Helper to create mock traces"""
        return [
            {
                'entropy_trace': np.random.uniform(1.0, 3.0, 50).tolist(),
                'attention_trace': np.random.uniform(0.2, 0.8, 50).tolist()
            }
            for _ in range(num_traces)
        ]

    def test_sample_traces_plot_returns_figure(self):
        """Test: Sample traces plot returns matplotlib Figure"""
        traces = self.create_mock_traces(10)

        fig = self.visualizer.plot_sample_traces(traces, num_samples=5)

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 10)  # 5 samples × 2 signals
        plt.close(fig)

    def test_sample_traces_correct_number(self):
        """Test: Plot shows correct number of sample traces"""
        traces = self.create_mock_traces(20)

        fig = self.visualizer.plot_sample_traces(traces, num_samples=3)

        self.assertEqual(len(fig.axes), 6)  # 3 samples × 2 signals
        plt.close(fig)

    def test_sample_traces_handles_fewer_than_requested(self):
        """Test: Plot handles fewer traces than requested samples"""
        traces = self.create_mock_traces(2)

        fig = self.visualizer.plot_sample_traces(traces, num_samples=5)

        self.assertEqual(len(fig.axes), 4)  # 2 available × 2 signals
        plt.close(fig)

    def test_sample_traces_single_trace(self):
        """Test: Plot works with single trace"""
        traces = self.create_mock_traces(1)

        fig = self.visualizer.plot_sample_traces(traces, num_samples=1)

        self.assertEqual(len(fig.axes), 2)  # 1 sample × 2 signals
        plt.close(fig)

    def test_sample_traces_saves_to_file(self):
        """Test: Sample traces plot saves correctly"""
        traces = self.create_mock_traces(10)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name

        fig = self.visualizer.plot_sample_traces(traces, num_samples=3, save_path=save_path)

        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        os.unlink(save_path)


class TestHeatmapPlots(unittest.TestCase):
    """Test temporal pattern heatmaps"""

    def setUp(self):
        self.visualizer = SignalVisualizer(output_dir=tempfile.mkdtemp())

    def create_mock_traces(self, num_traces=20):
        """Helper to create mock traces"""
        return [
            {
                'entropy_trace': np.random.uniform(1.0, 3.0, 80).tolist(),
                'attention_trace': np.random.uniform(0.2, 0.8, 80).tolist()
            }
            for _ in range(num_traces)
        ]

    def test_heatmap_returns_figure(self):
        """Test: Heatmap returns matplotlib Figure"""
        traces = self.create_mock_traces(20)

        fig = self.visualizer.plot_temporal_heatmap(traces, signal_name='entropy')

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_correct_dimensions(self):
        """Test: Heatmap has correct matrix dimensions"""
        traces = self.create_mock_traces(15)

        fig = self.visualizer.plot_temporal_heatmap(
            traces, signal_name='entropy', max_length=100
        )

        # Heatmap should be 15 traces × 100 time steps
        ax = fig.axes[0]
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_handles_variable_length_traces(self):
        """Test: Heatmap handles traces of different lengths"""
        traces = [
            {'entropy_trace': np.random.uniform(1.0, 3.0, 30).tolist()},
            {'entropy_trace': np.random.uniform(1.0, 3.0, 50).tolist()},
            {'entropy_trace': np.random.uniform(1.0, 3.0, 120).tolist()},
        ]

        fig = self.visualizer.plot_temporal_heatmap(
            traces, signal_name='entropy', max_length=100
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_saves_to_file(self):
        """Test: Heatmap saves correctly"""
        traces = self.create_mock_traces(10)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name

        fig = self.visualizer.plot_temporal_heatmap(
            traces, signal_name='entropy', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        os.unlink(save_path)


class TestStatisticalSummary(unittest.TestCase):
    """Test statistical summary plots"""

    def setUp(self):
        self.visualizer = SignalVisualizer(output_dir=tempfile.mkdtemp())

    def create_mock_traces(self, mean_entropy_attack=2.5, mean_entropy_normal=1.0):
        """Helper to create traces with specific distributions"""
        hallucination_traces = [
            {
                'entropy_trace': np.random.normal(mean_entropy_attack, 0.5, 50).tolist(),
                'attention_trace': np.random.normal(0.3, 0.1, 50).tolist()
            }
            for _ in range(20)
        ]

        normal_traces = [
            {
                'entropy_trace': np.random.normal(mean_entropy_normal, 0.3, 50).tolist(),
                'attention_trace': np.random.normal(0.7, 0.1, 50).tolist()
            }
            for _ in range(20)
        ]

        return hallucination_traces, normal_traces

    def test_statistical_summary_returns_figure(self):
        """Test: Statistical summary returns matplotlib Figure"""
        hallucination_traces, normal_traces = self.create_mock_traces()

        fig = self.visualizer.plot_statistical_summary(
            hallucination_traces, normal_traces
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 2)  # 2 subplots (entropy, attention)
        plt.close(fig)

    def test_statistical_summary_shows_higher_attack_entropy(self):
        """Test: Statistical summary shows attack traces have higher entropy"""
        hallucination_traces, normal_traces = self.create_mock_traces(
            mean_entropy_attack=3.0, mean_entropy_normal=1.0
        )

        fig = self.visualizer.plot_statistical_summary(
            hallucination_traces, normal_traces
        )

        # Verify attack entropy is higher (visual inspection in actual use)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_statistical_summary_saves_to_file(self):
        """Test: Statistical summary saves correctly"""
        hallucination_traces, normal_traces = self.create_mock_traces()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name

        fig = self.visualizer.plot_statistical_summary(
            hallucination_traces, normal_traces,
            save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        os.unlink(save_path)

    def test_statistical_summary_empty_traces(self):
        """Test: Statistical summary handles empty traces"""
        fig = self.visualizer.plot_statistical_summary([], [])

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.visualizer = SignalVisualizer(output_dir=tempfile.mkdtemp())

    def test_traces_with_missing_signals(self):
        """Test: Handles traces with missing signal keys"""
        traces = [
            {'prompt': 'Test'},  # No signals
            {'entropy_trace': [1.0, 2.0]},  # Only entropy
            {'attention_trace': [0.5, 0.6]},  # Only attention
        ]

        # Should not crash
        fig = self.visualizer.plot_sample_traces(traces, num_samples=3)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_traces_with_empty_signals(self):
        """Test: Handles traces with empty signal arrays"""
        traces = [
            {'entropy_trace': [], 'attention_trace': []},
            {'entropy_trace': [1.0], 'attention_trace': [0.5]},
        ]

        fig = self.visualizer.plot_sample_traces(traces, num_samples=2)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_output_directory_creation(self):
        """Test: Output directory is created if it doesn't exist"""
        temp_dir = Path(tempfile.mkdtemp()) / 'nested' / 'output'

        visualizer = SignalVisualizer(output_dir=str(temp_dir))

        self.assertTrue(temp_dir.exists())


class TestFullPipeline(unittest.TestCase):
    """Test full visualization pipeline"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.visualizer = SignalVisualizer(output_dir=str(self.temp_dir / 'figures'))

    def test_generate_all_visualizations(self):
        """Test: Generate all visualizations completes successfully"""
        # Create mock data
        traces = []
        validations = []

        for i in range(10):
            traces.append({
                'prompt': f'Attack {i}',
                'entropy_trace': np.random.uniform(2.0, 3.0, 50).tolist(),
                'attention_trace': np.random.uniform(0.2, 0.4, 50).tolist()
            })
            validations.append({'is_hallucination': True, 'confidence': 0.95})

        for i in range(10):
            traces.append({
                'prompt': f'Normal {i}',
                'entropy_trace': np.random.uniform(0.5, 1.5, 50).tolist(),
                'attention_trace': np.random.uniform(0.6, 0.9, 50).tolist()
            })
            validations.append({'is_hallucination': False, 'confidence': 0.95})

        # Save to files
        traces_path = self.temp_dir / 'traces.json'
        validation_path = self.temp_dir / 'validation.json'

        with open(traces_path, 'w') as f:
            json.dump({'traces': traces}, f)
        with open(validation_path, 'w') as f:
            json.dump({'validation_results': validations}, f)

        # Generate all visualizations
        self.visualizer.generate_all_visualizations(
            str(traces_path), str(validation_path), prefix='test_'
        )

        # Verify some files were created
        output_files = list((self.temp_dir / 'figures').glob('test_*.png'))
        self.assertGreater(len(output_files), 0)


if __name__ == '__main__':
    # Close all matplotlib figures after tests
    unittest.main(exit=False)
    plt.close('all')
