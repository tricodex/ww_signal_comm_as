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
        
        self.communication_summary = self.df['Communication'].describe()
        

        os.makedirs(self.output_dir, exist_ok=True)
        self.apply_dynamic_pca()
        self.apply_pca_to_dependent_vars()
        self.regression_on_principal_components()
        
        
    def apply_dynamic_pca(self, variance_threshold=0.9):
        pca = PCA().fit(self.scaled_features)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.sum(cumulative_variance < variance_threshold) + 1  # Adjust based on variance threshold
        print(f"Selected {n_components} components explaining at least {variance_threshold*100}% of variance.")
        
        pca_optimal = PCA(n_components=n_components)
        pca_components = pca_optimal.fit_transform(self.scaled_features)
        # Assign PCA components dynamically based on the number selected
        for i in range(n_components):
            self.df[f'PCA{i+1}'] = pca_components[:, i]

    def plot_cumulative_variance(self, pca):
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'cumulative_variance.png'))
        plt.close()




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
        print(result.summary())
        return result
    
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
        
    def calculate_individual_agent_entropy(self, signal_column='Communication'):
        """Calculates the entropy for each individual agent's signals."""
        entropies = {}
        for agent_id in self.unique_agents:
            agent_signals = self.df[self.df['AgentID'] == agent_id][signal_column]
            hist, bin_edges = np.histogram(agent_signals, bins=10, density=True)
            probabilities = hist * np.diff(bin_edges)
            entropy_value = entropy(probabilities[probabilities > 0], base=2)
            entropies[agent_id] = entropy_value
            print(f'Entropy for Agent {agent_id}: {entropy_value}')
        return entropies
    
    
    def summarize_and_calculate_entropy(self, column_index, n_bins=10):
        # Extract the column of signals based on the given index
        signals = self.actions_array[:, column_index]

        # Display statistical summary of the signals
        signals_series = pd.Series(signals)
        print("Statistical summary of the signals:")
        print(signals_series.describe())

        # Calculate entropy of the signals
        hist, bin_edges = np.histogram(signals, bins=n_bins, density=True)
        bin_probabilities = hist * np.diff(bin_edges)
        bin_probabilities = bin_probabilities[bin_probabilities > 0]
        entropy_value = entropy(bin_probabilities, base=2)

        # Display the calculated entropy
        print(f"\nEntropy of the signals: {entropy_value}")
        
        self.entropy_value = entropy_value
    
    def apply_hierarchical_clustering(self):
        # More comprehensive use of hierarchical clustering
        self.linked = linkage(self.scaled_features, method='ward')
        self.df['Hierarchical_Cluster'] = fcluster(self.linked, t=5, criterion='maxclust')
        

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
        for i in range(len(self.unique_agents)):
            for j in range(i + 1, len(self.unique_agents)):
                agent_i_signals = self.df[self.df['AgentID'] == self.unique_agents[i]]['Communication'].values
                agent_j_signals = self.df[self.df['AgentID'] == self.unique_agents[j]]['Communication'].values
                # Calculate mutual information only if both agents have emitted signals
                if len(agent_i_signals) > 0 and len(agent_j_signals) > 0:
                    mi = self.calculate_mutual_information(agent_i_signals, agent_j_signals)
                    self.mutual_info_results.append((self.unique_agents[i], self.unique_agents[j], mi))

    def print_mutual_info_results(self):
        for result in self.mutual_info_results:
            print(f'Mutual information between Agent {result[0]} and Agent {result[1]}: {result[2]}')

    
    

    def calculate_correlation_with_performance(self):
        communication_reward_correlation = self.df['Communication'].corr(self.df['Reward'])

        print(f"Correlation between Communication Signal and Performance: {communication_reward_correlation}")
        
        return communication_reward_correlation
               
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
        from scipy.cluster.hierarchy import fcluster
        self.df['Hierarchical_Cluster'] = fcluster(self.linked, n_clusters, criterion='maxclust')
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['Hierarchical_Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Hierarchical Cluster')
        plt.title(f'Hierarchical Clustering with {n_clusters} Clusters')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

       
    def perform_time_frequency_analysis(self, plot_name):
        frequencies, power_spectral_density = welch(self.df['Communication'], fs=1.0, window='hann', nperseg=1024, scaling='spectrum')
        plt.figure(figsize=(10, 6))
        plt.semilogy(frequencies, power_spectral_density)
        plt.title('Power Spectral Density of Communication Signals')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [V^2/Hz]')
        
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()
        
    def create_k_distance_plot(self, k=8, plot_name='k_distance_plot.png'):
        """
        Creates and saves a k-distance plot for choosing the 'eps' parameter in DBSCAN.
        Args:
        - k: Number of nearest neighbors to consider for the k-distance computation.
        - plot_name: The name of the plot image file to save.
        """
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
    

        
        





