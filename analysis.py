# analysis.py

# action_aray is a list of lists, each list contains 4 elements(actionspace+id): [HorizontalThrust, VerticalThrust, CommunicationSignal, Reward, AgentID]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
from scipy.signal import welch
import statsmodels.api as sm
import seaborn as sns
import os
from pandas.plotting import autocorrelation_plot

class Analysis:
    def __init__(self, actions_array, output_dir): 
        self.actions_array = actions_array
        #self.df = pd.DataFrame(actions_array, columns=['Horizontal', 'Vertical', 'Communication', 'AgentID'])
        self.df = pd.DataFrame(actions_array, columns=['Horizontal', 'Vertical', 'Communication', 'Reward', 'AgentID'])

        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[['Horizontal', 'Vertical', 'Communication']])
        self.pca = PCA(n_components=2)
        self.pca_components = self.pca.fit_transform(self.scaled_features)
        self.pca_df = pd.DataFrame(data=self.pca_components, columns=['PCA1', 'PCA2'])
        self.pca_df['AgentID'] = self.df['AgentID']
        self.X = sm.add_constant(self.df['Communication'])
        self.y = self.df['Horizontal']
        self.z = self.df['Vertical']
        self.model = sm.OLS(self.y, self.X).fit()
        self.df['Predicted_Horizontal'] = self.model.predict(self.X)
        self.residuals = self.model.resid
        self.kmeans = KMeans(n_clusters=4, random_state=0)
        self.df['Cluster'] = self.kmeans.fit_predict(self.scaled_features)
        
        self.regression_model = LinearRegression().fit(self.df[['Communication']], self.df['Horizontal'])
        self.df['Predicted_Horizontal'] = self.regression_model.predict(self.df[['Communication']])
        
        self.df['Movement_Magnitude'] = np.sqrt(self.df['Horizontal']**2 + self.df['Vertical']**2)
        self.regression_model = LinearRegression().fit(self.df[['Communication']], self.df['Movement_Magnitude'])
        self.df['Predicted_Movement_Magnitude'] = self.regression_model.predict(self.df[['Communication']])
        
        self.unique_agents = self.df['AgentID'].unique()
        self.mutual_info_results = []
        self.output_dir = output_dir  # Use this attribute to store outputs
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_mutual_information(self, signal1, signal2, n_bins=10):
        # Ensure signals have the same length by trimming to the shorter length
        min_length = min(len(signal1), len(signal2))
        signal1 = signal1[:min_length]
        signal2 = signal2[:min_length]
        
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
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
            
            
    
            

    def plot_autocorrelation(self, plot_name='autocorrelation_plot.png'):
        plt.figure(figsize=(10, 6))
        autocorrelation_plot(self.df['Communication'])
        plt.title('Autocorrelation of Communication Signal')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()


    def plot_movement_scatter(self, plot_name='movement_scatter_plot.png'):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['AgentID'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Agent ID')
        plt.title('Movement Scatter Plot Color-coded by Agent ID')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, plot_name))  
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
        
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close() 
        
    def plot_mutual_info_heatmap(self, plot_name='mutual_info_heatmap.png'):
        mi_matrix = np.zeros((len(self.unique_agents), len(self.unique_agents)))
        for result in self.mutual_info_results:
            i, j = int(result[0]), int(result[1])
            mi_matrix[i, j] = mi_matrix[j, i] = result[2]
        sns.heatmap(mi_matrix, annot=True, fmt=".2f", cmap="viridis")
        plt.title('Mutual Information Heatmap')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

        
    def plot_communication_over_time(self, plot_name='communication_over_time.png'):
        plt.figure(figsize=(10, 6))
        for agent_id in self.unique_agents:
            agent_data = self.df[self.df['AgentID'] == agent_id]
            plt.plot(agent_data.index, agent_data['Communication'], label=f'Agent {agent_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Communication Signal')
        plt.title('Communication Signal Over Time by Agent')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()


    def plot_pca_results(self, plot_name='pca_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.pca_df['PCA1'], self.pca_df['PCA2'], c=self.pca_df['AgentID'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Agent ID')
        plt.title('PCA: 2 Principal Components of Action Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

    def plot_clustering_results(self, plot_name='clustering_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Clustering: Agent Behavior Modeling')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

    def plot_residuals_vs_predicted(self, plot_name='residuals_vs_predicted_plot.png'):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Predicted_Horizontal'], self.residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Horizontal Movement')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Horizontal Movement')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()    

    def apply_dbscan(self, eps=0.5, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(self.scaled_features)
        self.df['DBSCAN_Cluster'] = self.dbscan.labels_

    def plot_dbscan_results(self, plot_name='dbscan_clustering_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['DBSCAN_Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='DBSCAN Cluster')
        plt.title('DBSCAN Clustering: Agent Behavior Modeling')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()

    def apply_hierarchical_clustering(self, method='ward'):
        self.linked = linkage(self.scaled_features, method=method)

        
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
        #frequencies, power_spectral_density = welch(self.df['Communication'], fs=1.0, window='hanning', nperseg=1024, scaling='spectrum')
        frequencies, power_spectral_density = welch(self.df['Communication'], fs=1.0, window='hann', nperseg=1024, scaling='spectrum')

        plt.figure(figsize=(10, 6))
        plt.semilogy(frequencies, power_spectral_density)
        plt.title('Power Spectral Density of Communication Signals')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [V^2/Hz]')
        
        plt.savefig(os.path.join(self.output_dir, plot_name))  
        plt.close()
        
        





