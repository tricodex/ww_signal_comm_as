import unittest
from unittest.mock import patch
import os
import numpy as np
from analysis import Analysis
import shutil
import sys
import datetime
from main import run_and_analyze_all_configs, compare_across_configurations, model_configs
import matplotlib.pyplot as plt


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)  # Ensuring reproducibility
        self.output_dir = 'test_output'
        actions_array = np.random.randint(1, 10, size=(1000, 5))
        self.analysis = Analysis(actions_array, self.output_dir)


    def tearDown(self):
        plt.close('all')  # Close all plots
        try:
            shutil.rmtree(self.output_dir)
        except PermissionError:
            print("PermissionError: Could not delete test output directory.")


    def test_apply_dynamic_pca(self):
        pca = self.analysis.apply_dynamic_pca(variance_threshold=0.9)
        # Ensure PCA has retained at least one component and check presence in dataframe
        self.assertGreaterEqual(pca.n_components_, 1)
        self.assertIn('PCA1', self.analysis.df.columns)

    def test_analyze_behavioral_impact(self):
        # The actual correlation will be tested for its presence rather than assuming a specific value
        correlation_horizontal, correlation_vertical = self.analysis.analyze_behavioral_impact()
        self.assertIsInstance(correlation_horizontal, float)
        self.assertIsInstance(correlation_vertical, float)

    def test_apply_pca_to_dependent_vars(self):
        self.analysis.apply_pca_to_dependent_vars()
        self.assertEqual(self.analysis.pca_df.shape[1], 2)

    def test_behavior_clustering(self):
        self.analysis.behavior_clustering()
        # Check that exactly 3 clusters were proposed
        self.assertEqual(self.analysis.df['BehaviorCluster'].nunique(), 3)

    def test_regression_on_principal_components(self):
        self.analysis.regression_on_principal_components()
        self.assertIsNotNone(self.analysis.model_pc1)
        self.assertIsNotNone(self.analysis.model_pc2)

    def test_save_results(self):
        self.analysis.save_results('behavior.txt')
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'behavior.txt')))

    def test_apply_dbscan(self):
        self.analysis.apply_dbscan()
        self.assertIsNotNone(self.analysis.dbscan)
        self.assertIn('DBSCAN_Cluster', self.analysis.df.columns)

    def test_apply_gee(self):
        gee_results = self.analysis.apply_gee()
        self.assertEqual(len(gee_results), 2)

    def test_generalized_estimating_equations(self):
        result = self.analysis.generalized_estimating_equations()
        self.assertIsNotNone(result)

    def test_plot_signal_histogram(self):
        self.analysis.plot_signal_histogram()
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'signal_histogram.png')))

    def test_calculate_individual_agent_entropy(self):
        entropies = self.analysis.calculate_individual_agent_entropy()
        self.assertEqual(len(entropies), len(self.analysis.unique_agents))

    def test_summarize_and_calculate_entropy(self):
        entropy_value = self.analysis.summarize_and_calculate_entropy(column_index=3, n_bins=5)
        self.assertIsNotNone(self.analysis.entropy_value)
        self.assertGreaterEqual(entropy_value, 0)

    def test_full_analysis(self):
        # This test ensures that the full analysis cycle can run without errors
        self.analysis.full_analysis()
        # Check that the results dictionary is populated correctly
        self.assertTrue('individual' in self.analysis.results)
        self.assertTrue('mutual_information' in self.analysis.results)
        self.assertTrue('PCA' in self.analysis.results)
        self.assertTrue('evaluation' in self.analysis.results)
        self.assertTrue('comparison' in self.analysis.results)

    def test_save_analysis_results(self):
        # Pre-populate results to simulate analysis outcomes
        self.analysis.results = {
            'individual': 'Simulated individual results',
            'mutual_information': 'Simulated mutual information',
            'PCA': 'Simulated PCA results'
        }
        self.analysis.save_analysis_results()
        # Check if the file has been created and contains the expected text
        results_path = os.path.join(self.output_dir, 'detailed_analysis_report.txt')
        self.assertTrue(os.path.exists(results_path))
        with open(results_path, 'r') as file:
            content = file.read()
            self.assertIn('Simulated individual results', content)
            self.assertIn('Simulated mutual information', content)
            self.assertIn('Simulated PCA results', content)

    def test_individual_analysis(self):
        # Perform individual analysis and check if the results are calculated correctly
        self.analysis.individual_analysis()
        self.assertTrue(isinstance(self.analysis.results['individual'], dict))

    def test_collective_analysis(self):
        # Perform collective analysis and check if the results are calculated correctly
        self.analysis.collective_analysis()
        self.assertIn('cluster', self.analysis.df.columns)
        self.assertIn('PCA1', self.analysis.df.columns)
        self.assertIn('PCA2', self.analysis.df.columns)

    def test_analysis_across_evaluations(self):
        # Perform analysis across evaluations and check if the results are calculated correctly
        self.analysis.analysis_across_evaluations()
        self.assertTrue(isinstance(self.analysis.results['evaluation'], dict))

    def test_comparative_analysis_across_configurations(self):
        # Simulate having multiple configurations
        self.analysis.results['evaluation'] = {
            'average_reward': 5,
            'average_entropy': 0.1,
            'average_mutual_information': 0.01
        }
        self.analysis.comparative_analysis_across_configurations()
        self.assertTrue(isinstance(self.analysis.results['comparison'], dict))
        
        
    @patch('main.eval_with_model_path_run')
    @patch('main.Analysis')
    def test_run_and_analyze_all_configs(self, mock_analysis, mock_eval):
        # Configure the mock to simulate behavior
        mock_eval.return_value = {'actions': np.random.randint(1, 10, size=(1000, 5)), 'overall_avg_reward': 0.5}
        mock_analysis_instance = mock_analysis.return_value
        mock_analysis_instance.results = {'evaluation': {'avg_reward': 0.5}}

        # Run the function with mocked behavior
        results = run_and_analyze_all_configs(games=10)

        # Assertions to check if analysis and file operations were called correctly
        self.assertTrue(mock_analysis.called)
        self.assertTrue(mock_eval.called)
        self.assertIn('PPO_pursuers_2', results)

if __name__ == '__main__':
    # Create the test_report directory if it doesn't exist
    report_dir = 'test_report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # Define the file path with current datetime for uniqueness
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(report_dir, f"{current_time}.txt")

    # Open the file to redirect the output of the tests
    with open(file_path, 'w') as f:
        # Create a test loader and suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        
        # Create a test runner that streams results to the file
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        # Run the tests
        runner.run(suite)
