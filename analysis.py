# analysis.py

# action_aray is a list of lists, each list contains 5 elements in a continious manner(actionspace+reward+id): [HorizontalThrust, VerticalThrust, CommunicationSignal, Reward, AgentID]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
from scipy.signal import welch

import seaborn as sns
import os

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence

from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable

from sklearn.neighbors import NearestNeighbors


class Analysis:
    def __init__(self, actions_array, output_dir):
        self.actions_array = actions_array
        self.output_dir = output_dir
        self.df = pd.DataFrame(actions_array, columns=['Horizontal', 'Vertical', 'Communication', 'Reward', 'AgentID'])
        self.scaled_features = StandardScaler().fit_transform(self.df[['Horizontal', 'Vertical', 'Communication']])
        self.unique_agents = self.df['AgentID'].unique()
        self.pca_df = None
        self.model_pc1 = None
        self.model_pc2 = None
        self.mutual_info_results = []
        self.dbscan = None
        self.linked = None
        self.entropy_value = None
        
        
        self.results = {}
        
        self.communication_summary = self.df['Communication'].describe()
        

        os.makedirs(self.output_dir, exist_ok=True)
        self.apply_dynamic_pca()
        self.apply_pca_to_dependent_vars()
        self.regression_on_principal_components()
        
        
    def apply_dynamic_pca(self, variance_threshold=0.9):
        """
        Applies PCA dynamically based on a variance threshold and ensures at least one component is selected.
        """
        pca = PCA().fit(self.scaled_features)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        valid_components = np.where(cumulative_variance >= variance_threshold)[0]

        if valid_components.size == 0:
            n_components = len(cumulative_variance)  # Use all components if none meet the threshold
            print("Warning: No components meet the variance threshold. Using all components.")
        else:
            n_components = valid_components[0] + 1

        print(f"Selected {n_components} components, explaining at least {variance_threshold*100:.2f}% of variance.")
        
        pca_optimal = PCA(n_components=n_components)
        pca_components = pca_optimal.fit_transform(self.scaled_features)
        self.df[[f'PCA{i+1}' for i in range(n_components)]] = pca_components
        return pca_optimal


    
        
    def analyze_behavioral_impact(self):
        """
        Analyzes the impact of communication on movement by examining correlations.
        """
        # Calculate changes in movement
        self.df['HorizontalChange'] = self.df['Horizontal'].diff().fillna(0)
        self.df['VerticalChange'] = self.df['Vertical'].diff().fillna(0)
        
        # Calculate correlations
        correlation_horizontal = self.df['Communication'].corr(self.df['HorizontalChange'])
        correlation_vertical = self.df['Communication'].corr(self.df['VerticalChange'])
        
        print(f"Correlation between Communication and Horizontal Movement Change: {correlation_horizontal:.3f}")
        print(f"Correlation between Communication and Vertical Movement Change: {correlation_vertical:.3f}")

        return correlation_horizontal, correlation_vertical

    def apply_pca_to_dependent_vars(self):
        dependent_vars = self.df[['Horizontal', 'Vertical']]
        dependent_vars_scaled = StandardScaler().fit_transform(dependent_vars)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(dependent_vars_scaled)
        self.pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    def behavior_clustering(self, plot_name='behavior_clustering.png'):
        pca_result = self.scaled_features
        
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=0).fit(pca_result)
        self.df['BehaviorCluster'] = kmeans.labels_
        
        # Visualize clusters
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.df['BehaviorCluster'])
        plt.title('Agent Behavior Clustering')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()
  
    def regression_on_principal_components(self):
        X = sm.add_constant(self.df['Communication'])
        self.model_pc1 = sm.OLS(self.pca_df['PC1'], X).fit()
        self.model_pc2 = sm.OLS(self.pca_df['PC2'], X).fit()


    def save_results(self, filename='behavior.txt'):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            
                # Categorizing rewards
                self.df['Interaction'] = self.df['Reward'].apply(lambda x: 'Food' if x > (69/len(self.unique_agents)) else ('Poison' if x < (-9.9/len(self.unique_agents)) else 'Neutral'))

                # Aggregating results by AgentID and Interaction
                interaction_counts = self.df.groupby(['AgentID', 'Interaction']).size().unstack(fill_value=0)
                f.write("Interaction Counts:\n" + interaction_counts.to_string() + "\n\n")

                # Select only numeric columns from the DataFrame for rolling operation
                numeric_df = self.df.select_dtypes(include=[np.number])

                # Apply the rolling window and sum operations on this numeric-only DataFrame
                cooperative_moves = numeric_df[numeric_df['Reward'] > 0].groupby('AgentID').rolling(window=2).sum()

                # Filter to get consecutive positive rewards
                cooperative_moves = cooperative_moves[cooperative_moves['Reward'] > (69/len(self.unique_agents))]
                f.write("Cooperative Moves:\n" + cooperative_moves.to_string() + "\n")
        

    def save_fig(self, fig, plot_name):
        """Saves matplotlib figures in the designated output directory."""
        fig.savefig(os.path.join(self.output_dir, plot_name))
        plt.close(fig)
        
    
    def apply_dbscan(self, eps=0.5, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(self.scaled_features)
        self.df['DBSCAN_Cluster'] = self.dbscan.labels_
        
    def apply_gee(self):
        # Define the independent and dependent variables
        indep_vars = self.df[['Communication']]
        dep_vars = self.df[['Horizontal', 'Vertical']]
        
        # Add a constant to the independent variables for the intercept
        indep_vars = sm.add_constant(indep_vars)
        
        # Create a GEE model for each dependent variable
        gee_results = {}
        for dep_var_name in dep_vars.columns:
            model = GEE(self.df[dep_var_name], indep_vars, groups=self.df['AgentID'], family=Gaussian(), cov_struct=Exchangeable())
            results = model.fit()
            gee_results[dep_var_name] = results
        
        return gee_results
   
    def generalized_estimating_equations(self):
        # GEE model for correlated movements within agents over time
        model = GEE.from_formula("Horizontal + Vertical ~ Communication", "AgentID", self.df, cov_struct=Independence())
        result = model.fit()
        #print(result.summary())
        return result
    
    
        
    def calculate_individual_agent_entropy(self, signal_column='Communication'):
        """Calculates the entropy for each individual agent's signals."""
        entropies = {}
        for agent_id in self.unique_agents:
            agent_signals = self.df[self.df['AgentID'] == agent_id][signal_column]
            hist, bin_edges = np.histogram(agent_signals, bins=10, density=True)
            probabilities = hist * np.diff(bin_edges)
            entropy_value = entropy(probabilities[probabilities > 0], base=2)
            entropies[agent_id] = entropy_value
            #print(f'Entropy for Agent {agent_id}: {entropy_value}')
        return entropies
    
    
    def summarize_and_calculate_entropy(self, column_index, n_bins=10):
        # Extract the column of signals based on the given index
        signals = self.actions_array[:, column_index]

        # Display statistical summary of the signals
        signals_series = pd.Series(signals)
        #print("Statistical summary of the signals:")
        #print(signals_series.describe())

        # Calculate entropy of the signals
        hist, bin_edges = np.histogram(signals, bins=n_bins, density=True)
        bin_probabilities = hist * np.diff(bin_edges)
        bin_probabilities = bin_probabilities[bin_probabilities > 0]
        entropy_value = entropy(bin_probabilities, base=2)

        # Display the calculated entropy
        print(f"\nEntropy of the signals: {entropy_value}")
        
        self.entropy_value = entropy_value
        
        return entropy_value
    
    def apply_hierarchical_clustering(self):
        # Ensure that scaled features are in floating-point format suitable for hierarchical clustering
        if self.scaled_features.dtype != np.float64:
            self.scaled_features = self.scaled_features.astype(np.float64)

        # More comprehensive use of hierarchical clustering
        self.linked = linkage(self.scaled_features, method='ward')

        

    def calculate_mutual_information(self, signal1, signal2):
        # Ensure signals have the same length by trimming to the shorter length
        min_length = min(len(signal1), len(signal2))
        signal1 = signal1[:min_length]
        signal2 = signal2[:min_length]
        
        est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        signal1_discretized = est.fit_transform(signal1.reshape(-1, 1)).flatten()
        signal2_discretized = est.fit_transform(signal2.reshape(-1, 1)).flatten()
        return mutual_info_score(signal1_discretized, signal2_discretized)
    
    def calculate_mutual_info_results(self):
        results = []
        for i in range(len(self.unique_agents)):
            for j in range(i + 1, len(self.unique_agents)):
                agent_i_signals = self.df[self.df['AgentID'] == self.unique_agents[i]]['Communication']
                agent_j_signals = self.df[self.df['AgentID'] == self.unique_agents[j]]['Communication']
                
                if len(agent_i_signals) > 0 and len(agent_j_signals) > 0:
                    mi = self.calculate_mutual_information(agent_i_signals.values, agent_j_signals.values)
                    results.append({'agents': (self.unique_agents[i], self.unique_agents[j]), 'MI': mi})
                    #print(f'Mutual information between Agent {self.unique_agents[i]} and Agent {self.unique_agents[j]}: {mi:.3f}')
        print(f"MI for {len(results)} pairs, succes!")
        self.mutual_info_results = results
        return results

    

    def calculate_correlation_with_performance(self):
        communication_reward_correlation = self.df['Communication'].corr(self.df['Reward'])

        print(f"Correlation between Communication Signal and Performance: {communication_reward_correlation}")
        
        return communication_reward_correlation
               

        
    def full_analysis(self):
        """ Conduct the full multi-layer analysis as per the structured approach. """
        
        self.individual_analysis()
        
        self.collective_analysis()
        
        self.analysis_across_evaluations()
        
        self.save_analysis_results()
        
    def save_analysis_results(self):
        """ Save results in a human-readable format, with error handling for missing data. """
        with open(os.path.join(self.output_dir, 'detailed_analysis_report.txt'), 'w') as file:
            # Safely write individual entropy results
            individual_results = self.results.get('individual', 'No individual analysis results available.')
            file.write(f"Individual Entropy Results: {individual_results}\n")

            # Safely write mutual information results
            mutual_info_results = self.results.get('mutual_information', 'No mutual information results available.')
            file.write(f"Mutual Information: {mutual_info_results}\n")

            # Safely write PCA results
            pca_results = self.results.get('PCA', 'No PCA results available.')
            file.write(f"PCA Results: {pca_results}\n")


    def individual_analysis(self):
        """ Perform individual-level analysis for each agent. """
        self.results['individual'] = {}
        for agent_id in self.df['AgentID'].unique():
            agent_data = self.df[self.df['AgentID'] == agent_id]
            entropy_value = self.calculate_entropy(agent_data['Communication'])
            correlation = agent_data['Communication'].corr(agent_data['Reward'])
            self.results['individual'][agent_id] = {
                'entropy': entropy_value,
                'correlation': correlation
            }

    def collective_analysis(self):
        """ Perform collective-level analysis over all agents. """
        # Mutual Information
        self.results['mutual_information'] = self.calculate_mutual_info_results()

        # Clustering
        kmeans = KMeans(n_clusters=3).fit(self.scaled_features)
        self.df['cluster'] = kmeans.labels_

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.scaled_features)
        self.df['PCA1'], self.df['PCA2'] = pca_result[:, 0], pca_result[:, 1]
        self.results['PCA'] = pca.explained_variance_ratio_


    def analysis_across_evaluations(self):
        """ Analyze results across 100 evaluations for a single configuration. """
        # Assuming evaluations are somehow distinguished or aggregated in self.df
        self.results['evaluation'] = {
            'average_reward': self.df['Reward'].mean(),
            'average_entropy': np.mean([self.results['individual'][aid]['entropy'] for aid in self.df['AgentID'].unique()]),
            'average_mutual_information': self.results['mutual_information']
        }

    

    def calculate_entropy(self, signals):
        """ Calculate entropy of communication signals. """
        hist, bin_edges = np.histogram(signals, bins=10, density=True)
        prob_density = hist * np.diff(bin_edges)
        return entropy(prob_density[prob_density > 0])
    
    
    def plot_signal_histogram(self, plot_name='signal_histogram.png'):
        # Extract the signal column
        signal = self.df['Communication'].values
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(signal, bins=10, density=True, alpha=0.6, color='g')
        # Add titles and labels
        plt.title('Histogram of Communication Signals')
        plt.xlabel('Signal Value')
        plt.ylabel('Probability Density')
        # Save the histogram figure
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()    
        
    def plot_cumulative_variance(self):
        """
        Plots the cumulative variance explained by the principal components to aid in deciding how many components to retain.
        """
        pca = PCA().fit(self.scaled_features)
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'cumulative_variance.png'))
        plt.close()
    
    
    
    def plot_movement_communication_scatter(self, plot_name='movement_communication_scatter_plot.png'):
        # Normalizing the Communication signal for better visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(1, 10))  # Scale between 1 and 10 for marker sizes
        communication_scaled = scaler.fit_transform(self.df[['Communication']])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use scaled communication signal for marker size
        scatter = ax.scatter(self.df['Horizontal'], self.df['Vertical'], self.df['Communication'],
                            c=self.df['AgentID'], cmap='viridis', alpha=0.5, 
                            s=communication_scaled.flatten())  # Adjust marker size

        fig.colorbar(scatter, ax=ax, label='Agent ID')
        ax.set_title('3D Scatter Plot of Movements and Communication Signal, Color-coded by Agent ID')
        ax.set_xlabel('Horizontal Movement')
        ax.set_ylabel('Vertical Movement')
        ax.set_zlabel('Communication Signal')
        
        self.save_fig(fig, plot_name)
        
    def plot_mutual_info_heatmap(self, plot_name='mutual_info_heatmap.png'):
        plt.figure(figsize=(10, 6))
        mi_matrix = np.zeros((len(self.unique_agents), len(self.unique_agents)))
        for result in self.mutual_info_results:
            i, j = int(result[0]), int(result[1])
            mi_matrix[i, j] = mi_matrix[j, i] = result[2]
        sns.heatmap(mi_matrix, annot=True, fmt=".2f", cmap="viridis")
        plt.title('Mutual Information Heatmap')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')
        plt.tight_layout()  
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

    def plot_dbscan_results(self, plot_name='dbscan_clustering_plot.png'):
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['DBSCAN_Cluster'], cmap='viridis', alpha=0.5)
        fig.colorbar(scatter, ax=ax, label='DBSCAN Cluster')
        ax.set_title('DBSCAN Clustering: Agent Behavior Modeling')
        ax.set_xlabel('Horizontal Movement')
        ax.set_ylabel('Vertical Movement')
        self.save_fig(fig, plot_name)  # Correctly pass the figure object


    def plot_hierarchical_clusters(self, n_clusters, plot_name='hierarchical_clustering_plot.png'):
        if hasattr(self, 'linked') and self.linked is not None:
            clusters = fcluster(self.linked, t=n_clusters, criterion='maxclust')
            self.df['Hierarchical_Cluster'] = clusters

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['Hierarchical_Cluster'], cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, label='Hierarchical Cluster')
            plt.title(f'Hierarchical Clustering with {n_clusters} Clusters')
            plt.xlabel('Horizontal Movement')
            plt.ylabel('Vertical Movement')
            plt.savefig(os.path.join(self.output_dir, plot_name))
            plt.close()
        else:
            print("Linkage matrix not found. Ensure that 'apply_hierarchical_clustering' is called before plotting.")


    def create_k_distance_plot(self, k=8, plot_name='k_distance_plot.png'):
    
        # Compute the k-nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.scaled_features)
        distances, indices = nbrs.kneighbors(self.scaled_features)

        # Sort the distances
        k_dist_sorted = np.sort(distances[:, k-1], axis=0)

        # Plot the k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_dist_sorted)), k_dist_sorted)
        plt.title(f"k-Distance Plot (k={k})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {k}-th nearest neighbor")
        plt.grid(True)

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()

    def perform_time_frequency_analysis(self, plot_name='time_frequency_analysis.png'):
        from scipy.signal import spectrogram
        signal = self.df['Communication'].values
        f, t, Sxx = spectrogram(signal, fs=1)  # Assuming 1 Hz sampling rate; adjust as necessary
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')
        plt.title('Spectrogram of Communication Signal')
        plt.colorbar(label='Intensity (dB)')
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()

    
    

        
        





