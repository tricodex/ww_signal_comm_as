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


class Analysis:
    def __init__(self, actions_array, output_dir):
        self.mutual_info_results = []
        self.actions_array = actions_array
        
        # Initialize DataFrame with specified columns
        self.df = pd.DataFrame(actions_array, columns=['Horizontal', 'Vertical', 'Communication', 'Reward', 'AgentID'])
        
        # Apply standard scaling to the relevant features
        self.scaled_features = StandardScaler().fit_transform(self.df[['Horizontal', 'Vertical', 'Communication']])
        
        # Apply PCA based on explained variance ratio
        self.apply_dynamic_pca()
        
        # Preparing for regression analysis
        self.X = sm.add_constant(self.df['Communication'])  # Add constant term for intercept
        
        # Apply multivariate OLS regression
        self.apply_multivariate_OLS()
        
        # Perform clustering
        self.apply_kmeans_clustering()
        
        self.unique_agents = self.df['AgentID'].unique()
        
        # Prepare the output directory for saving results
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_fig(self, fig, plot_name):
        """Saves matplotlib figures in the designated output directory."""
        fig.savefig(os.path.join(self.output_dir, plot_name))
        plt.close(fig)
        
    def analyze_rewards(self):
        # Categorizing rewards
        self.df['Interaction'] = self.df['Reward'].apply(lambda x: 'Food' if x > 69 else ('Poison' if x < -9.9 else 'Neutral'))

        # Aggregating results by AgentID
        interaction_counts = self.df.groupby(['AgentID', 'Interaction']).size().unstack(fill_value=0)
        print(interaction_counts)
        
    def cooperative_analysis(self):
        # Assuming cooperation if multiple agents get positive rewards consecutively
        cooperative_moves = self.df[self.df['Reward'] > 0].groupby('AgentID').rolling(window=2).sum()
        cooperative_moves = cooperative_moves[cooperative_moves['Reward'] > 69] # Filter to get consecutive positive rewards
        print(cooperative_moves)
        
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

    def apply_dynamic_pca(self):
        """Applies PCA to the scaled features, selecting the number of components based on cumulative explained variance."""
        pca = PCA().fit(self.scaled_features)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= 0.9)[0][0] + 1  # Choose components to explain 90% of variance
        
        # Refit PCA with optimal number of components determined
        pca_optimal = PCA(n_components=n_components)
        pca_components_optimal = pca_optimal.fit_transform(self.scaled_features)
        
        # Update the DataFrame with the first two PCA components for potential visualization
        for i in range(n_components):
            self.df[f'PCA{i+1}'] = pca_components_optimal[:, i]

    def apply_multivariate_OLS(self):
        """Performs separate OLS regression for horizontal and vertical movements against communication signal."""
        # For simplicity and clarity, handling the horizontal movement
        y_horiz = self.df['Horizontal']
        model_horiz = sm.OLS(y_horiz, self.X).fit()
        
        # Handling the vertical movement
        y_vert = self.df['Vertical']
        model_vert = sm.OLS(y_vert, self.X).fit()
        
        # Storing predictions for visualization or further analysis
        self.df['Predicted_Horizontal'] = model_horiz.predict(self.X)
        self.df['Predicted_Vertical'] = model_vert.predict(self.X)
        
        # Compute residuals for horizontal regression for potential diagnostic analysis
        self.residuals = model_horiz.resid

    def apply_kmeans_clustering(self):
        """Applies KMeans clustering to the PCA components."""
        # Assuming two principal components for KMeans input (adjust based on actual PCA application)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(self.df[['PCA1', 'PCA2']])
        self.df['Cluster'] = kmeans.labels_
        
    def integrate_analysis_methods(self):
        # Apply Multivariate Regression
        self.model = self.multivariate_OLS()  # Assumes this method is updated for multivariate regression
        
        # Compute Movement Magnitude for each agent
        self.df['Movement_Magnitude'] = np.sqrt(self.df['Horizontal']**2 + self.df['Vertical']**2)
        
        self.df['Predicted_Movement_Magnitude'] = self.model.predict(self.df[['Communication']])  

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
   
    def multivariate_OLS(self):
        X = self.df[['Communication']]  # Independent variables
        y = self.df[['Horizontal', 'Vertical']]  # Dependent variables
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        multivariate_model = sm.OLS(y, X).fit()
        print(multivariate_model.summary())
        return multivariate_model
    
    def generalized_estimating_equations(self):
        # GEE model for correlated movements within agents over time
        model = GEE.from_formula("Horizontal + Vertical ~ Communication", "AgentID", self.df, cov_struct=Independence())
        result = model.fit()
        print(result.summary())
        return result
    
    
    def calculate_entropy(self, signal):
        # Calculation of entropy with error handling
        if np.all(signal == 0) or len(np.unique(signal)) == 1:
            return 0
        signal_probabilities = signal / signal.sum()
        return entropy(signal_probabilities, base=2)
    
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

    def save_mutual_info_results(self, filename='mutual_information_results.txt'):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for result in self.mutual_info_results:
                f.write(f'Mutual information between Agent {result[0]} and Agent {result[1]}: {result[2]}\n')

                
    def save_analysis_results(self, filename='analysis_results.txt'):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(str(self.model.summary()) + '\n')

            jb_test = sm.stats.stattools.jarque_bera(self.residuals)
            f.write(f"Jarque-Bera test statistic: {jb_test[0]}, p-value: {jb_test[1]}\n")

            bp_test = sm.stats.diagnostic.het_breuschpagan(self.residuals, self.model.model.exog)
            f.write(f"Breusch-Pagan test statistic: {bp_test[0]}, p-value: {bp_test[1]}\n")

            signal_entropy = entropy(self.df['Communication'])
            f.write(f'Entropy of Communication Signals: {signal_entropy}\n')

    def calculate_correlation_with_performance(self):
        communication_reward_correlation = self.df['Communication'].corr(self.df['Reward'])

        print(f"Correlation between Communication Signal and Performance: {communication_reward_correlation}")
        correlation_result_path = os.path.join(self.output_dir, 'correlation_analysis.txt')
        with open(correlation_result_path, 'w') as f:
            f.write(f"Correlation between Communication Signal and Performance: {communication_reward_correlation}\n")
            
               
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


    def plot_pca_results(self, plot_name='pca_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['PCA1'], self.df['PCA2'], c=self.df['AgentID'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Agent ID')
        plt.title('PCA: 2 Principal Components of Action Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        self.save_fig(plt, plot_name)  # Ensure save_fig method uses plt and not scatter for saving the figure.


    def plot_clustering_results(self, plot_name='clustering_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Clustering: Agent Behavior Modeling')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        self.save_fig(scatter, plot_name)

    def plot_dbscan_results(self, plot_name='dbscan_clustering_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['DBSCAN_Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='DBSCAN Cluster')
        plt.title('DBSCAN Clustering: Agent Behavior Modeling')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        self.save_fig(scatter, plot_name)

    def plot_hierarchical_clusters(self, n_clusters, plot_name='hierarchical_clustering_plot.png'):
        from scipy.cluster.hierarchy import fcluster
        self.df['Hierarchical_Cluster'] = fcluster(self.linked, n_clusters, criterion='maxclust')
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['Hierarchical_Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Hierarchical Cluster')
        plt.title(f'Hierarchical Clustering with {n_clusters} Clusters')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        self.save_fig(scatter, plot_name)

       
    def perform_time_frequency_analysis(self, plot_name):
        frequencies, power_spectral_density = welch(self.df['Communication'], fs=1.0, window='hann', nperseg=1024, scaling='spectrum')
        plt.figure(figsize=(10, 6))
        plt.semilogy(frequencies, power_spectral_density)
        plt.title('Power Spectral Density of Communication Signals')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [V^2/Hz]')
        
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()
        
    def plot_communication_over_time(self, plot_name='communication_over_time.png'):
        plt.figure(figsize=(12, 8))
        # Sample a subset of agents if there are too many
        sampled_agents = np.random.choice(self.unique_agents, min(len(self.unique_agents), 10), replace=False)
        for agent_id in sampled_agents:
            agent_data = self.df[self.df['AgentID'] == agent_id]
            # Consider averaging communication signals over a window to smooth out the data
            rolling_mean = agent_data['Communication'].rolling(window=10, min_periods=1).mean()
            plt.plot(agent_data.index, rolling_mean, label=f'Agent {agent_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Communication Signal (Rolling Mean)')
        plt.title('Communication Signal Over Time by Agent (Sampled & Smoothed)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

        
        





