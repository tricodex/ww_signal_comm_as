# analysis.py

# action_aray is a list of lists, each list contains 'Horizontal' (thrust), 'Vertical' (thrust), 'Signal', 'XPosition', 'YPosition', 'Reward', 'AgentID' in a continious manner.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import spectrogram
from sklearn.cluster import SpectralClustering




class Analysis:
    def __init__(self, actions_array, output_dir):
        self.actions_array = actions_array
        self.output_dir = output_dir
        self.df = pd.DataFrame(self.actions_array, columns=['Horizontal', 'Vertical', 'Signal', 'XPosition', 'YPosition', 'Reward', 'AgentID'])
        self.scaled_features = StandardScaler().fit_transform(self.df[['Horizontal', 'Vertical', 'Signal']])
        self.unique_agents = self.df['AgentID'].unique()
        self.pca_df = None
        self.model_pc = None
        self.model_pc1 = None
        self.model_pc2 = None
        self.mutual_info_results = []
        self.dbscan = None
        self.linked = None
        self.entropy_value = None
        self.results = {}
        self.communication_summary = self.df['Signal'].describe()
        os.makedirs(self.output_dir, exist_ok=True)
        self.create_features()
        self.path_efficiency_analysis()
        self.apply_dynamic_pca()
        self.plot_correlation_matrix()
        
        
    def create_features(self):
        # Feature Engineering
        self.df['ThrustInteraction'] = self.df['Horizontal'] * self.df['Vertical']  # Adding interaction term
        self.df['Radius'] = np.sqrt(self.df['XPosition']**2 + self.df['YPosition']**2)
        self.df['Angle'] = np.arctan2(self.df['YPosition'], self.df['XPosition'])
        self.df['Velocity_X'] = self.df['XPosition'].diff()  # Difference from the previous timestep
        self.df['Velocity_Y'] = self.df['YPosition'].diff()
        # Signal
        self.df['SignalChange'] = self.df['Signal'].diff()
        self.df['Signal_RollingMean'] = self.df['Signal'].rolling(window=5).mean()
        self.df['Signal_RollingStd'] = self.df['Signal'].rolling(window=5).std()
        self.df['Signal_ThrustInteraction'] = self.df['Signal'] * (self.df['Horizontal'] + self.df['Vertical'])
        self.df['PHighSignal'] = (self.df['Signal'] > 0.5).astype(float)
        self.df['NHighSignal'] = (self.df['Signal'] < -0.5).astype(float)
        # Categorize rewards
        self.df['RewardCategory'] = self.df['Reward'].apply(
            lambda x: 'Food' if x > (69/len(self.unique_agents)) else ('Poison' if x < (-9.9/len(self.unique_agents)) else 'Neutral')
        )
        
        

    


    def cluster(self, method='dbscan', n_clusters=3, affinity='rbf'):
        if method == 'kmeans':
            self.df.fillna(0, inplace=True) # Fill missing values with mean
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            self.df['RewardCategory'] = le.fit_transform(self.df['RewardCategory'])
            self.df['km'] = KMeans(n_clusters=n_clusters).fit_predict(self.df[['ThrustInteraction', 'Signal_RollingMean', 'RewardCategory']])
        elif method == 'spectral':
            self.df['spc'] = SpectralClustering(n_clusters=n_clusters, affinity=affinity).fit_predict(self.df[['Signal', 'Signal_ThrustInteraction']])
        else:
            print("Invalid clustering method specified.")
        return self.df

    def plot_clusters(self):
        for col in ['db', 'km', 'spc']:
            if col in self.df.columns:
                # Reduce dimensions with PCA if needed
                if self.df[col].nunique() > 10:
                    print(f"Reducing dimensionality for plotting {col}...")
                    df_temp = self.apply_dynamic_pca()  # Assuming you have this function
                else:
                    df_temp = self.df.copy()

                plt.figure(figsize=(8, 6))
                plt.scatter(df_temp['XPosition'], df_temp['YPosition'], c=df_temp[col], cmap='viridis', s=50) 
                plt.title(f"Clustering Results ({col})")
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.savefig(os.path.join(self.output_dir, f"{col}.png"))
                plt.close()  





        
    
                
    def calculate_vif(self):
        # Extract features and handle any potential infinite or NaN values appropriately
        features = self.df[['Signal', 'ThrustInteraction', 'Velocity_X', 'Velocity_Y']].dropna().replace([np.inf, -np.inf], np.nan)
        # Initialize DataFrame for VIF results
        vif_data = pd.DataFrame()
        vif_data['VIF_Factor'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
        vif_data['Feature'] = features.columns
        return vif_data
    
    
        
    def perform_comprehensive_analysis(self):
        
        vif_data = self.calculate_vif()
        
        spearman_corr, spearman_p = spearmanr(self.df.dropna(subset=['Signal', 'Reward'])['Signal'], self.df.dropna(subset=['Signal', 'Reward'])['Reward'])
        
        self.df['IsFood'] = (self.df['RewardCategory'] == 'Food').astype(int)


        # GLM for binary 'IsFood' based on 'Signal' and 'ThrustInteraction'
        model = sm.GLM.from_formula('IsFood ~ Signal + ThrustInteraction', family=sm.families.Binomial(), data=self.df)
        glm_results = model.fit()
        

        # Save all results to a text file
        with open(os.path.join(self.output_dir, 'stats_results.txt'), 'w') as f:
            f.write(vif_data.to_string() + '\n')
            f.write(f"Spearman Correlation between Signal and Reward: {spearman_corr}, p-value: {spearman_p}\n")
            f.write(glm_results.summary().as_text() + '\n')

        
    def analyze_rewards_correlation(self):
        

        # Group by reward category to analyze different metrics
        grouped = self.df.groupby('RewardCategory').mean()

        # Correlation of average signal and movement speed per category
        plt.figure(figsize=(10, 5))
        sns.barplot(x=grouped.index, y=grouped['Signal'], palette='viridis')
        plt.title('Average Signal Intensity by Reward Category')
        plt.ylabel('Average Signal Intensity')
        plt.savefig(f"{self.output_dir}/signal_reward_category_analysis.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(x=grouped.index, y=np.sqrt(grouped['Velocity_X']**2 + grouped['Velocity_Y']**2), palette='viridis')
        plt.title('Average Movement Speed by Reward Category')
        plt.ylabel('Average Movement Speed')
        plt.savefig(f"{self.output_dir}/velocity_reward_category_analysis.png")
        plt.close()
        
    def cluster_by_reward_category(self):
        for category in self.df['RewardCategory'].unique():
            cat_data = self.df[self.df['RewardCategory'] == category].copy()
            features = cat_data[['Velocity_X', 'Velocity_Y', 'Signal']]
            
            # Handling NaN values
            features.fillna(0, inplace=True)  
            
            if not features.empty:
                clustering = KMeans(n_clusters=3).fit(features)
                cat_data.loc[features.index, 'Cluster'] = clustering.labels_

                plt.figure()
                sns.scatterplot(x='Velocity_X', y='Velocity_Y', hue='Cluster', data=cat_data)
                plt.title(f'Clustering of Movements in {category}')
                plt.savefig(f"{self.output_dir}/cluster_{category}.png")
                plt.close()

            
    def dynamic_behavior_before_encounters(self):
        # Find indices of food and poison encounters
        food_indices = self.df[self.df['RewardCategory'] == 'Food'].index
        poison_indices = self.df[self.df['RewardCategory'] == 'Poison'].index

        # Analyze behavior some steps before the encounter
        lookback_steps = 10  # Analyze 5 steps before an encounter

        for index in food_indices:
            if index > lookback_steps:
                data_segment = self.df.loc[index-lookback_steps:index]
                
                plt.plot(data_segment['Velocity_X'], label='Velocity X')
                plt.plot(data_segment['Velocity_Y'], label='Velocity Y')
                plt.title('Agent Behavior Before Food Encounter')
                plt.legend()
                plt.savefig(f"{self.output_dir}/food_behavior_example.png")
                plt.close()
                break  

        for index in poison_indices:
            if index > lookback_steps:
                data_segment = self.df.loc[index-lookback_steps:index]
            
                plt.plot(data_segment['Velocity_X'], label='Velocity X')
                plt.plot(data_segment['Velocity_Y'], label='Velocity Y')
                plt.title('Agent Behavior Before Poison Encounter')
                plt.legend()
                plt.savefig(f"{self.output_dir}/poison_behavior_example.png")
                plt.close()
                break  
            
    def signal_behavior_before_encounters(self):
        # Find indices of food and poison encounters
        food_indices = self.df[self.df['RewardCategory'] == 'Food'].index
        poison_indices = self.df[self.df['RewardCategory'] == 'Poison'].index

        lookback_steps = 100 # Analyze 5 steps before an encounter

        for index in food_indices:
            if index > lookback_steps:
                data_segment = self.df.loc[index-lookback_steps:index]
                plt.plot(data_segment['Signal'], label='Signal')
                
                plt.title('Agent Behavior Before Food Encounter')
                plt.legend()
                plt.savefig(f"{self.output_dir}/signal_food_behavior_example.png")
                plt.close()
                break  

        for index in poison_indices:
            if index > lookback_steps:
                data_segment = self.df.loc[index-lookback_steps:index]
                plt.plot(data_segment['Signal'], label='Signal')
                
                plt.title('Agent Behavior Before Poison Encounter')
                plt.legend()
                plt.savefig(f"{self.output_dir}/signal_poison_behavior_example.png")
                plt.close()
                break  
            
    def plot_time_series_analysis(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        self.df[['Signal']].plot(ax=axs[0])
        axs[0].set_title('Signal Over Time')
        self.df[['Velocity_X', 'Velocity_Y']].plot(ax=axs[1])
        axs[1].set_title('Velocity Components Over Time')
        self.df[['ThrustInteraction']].plot(ax=axs[2])
        axs[2].set_title('Thrust Interaction Over Time')
        plt.xlabel('Time Steps')
        plt.savefig(f"{self.output_dir}/time_series_analysis.png")
        plt.close()
        
    def regression_with_interactions(self):
        # Adding interaction terms
        X = self.df[['Horizontal', 'Vertical', 'Signal', 'Signal_ThrustInteraction', 'Radius', 'Angle']]
        y = self.df['Reward']
        X = sm.add_constant(X)  # Adding a constant for the intercept
        model = sm.OLS(y, X).fit()
        with open(f"{self.output_dir}/regression_with_interactions_summary.txt", 'w') as file:
            file.write(model.summary().as_text())


    
    
    def plot_feature_correlation_matrix(self):
        correlation_matrix = self.df[['Horizontal', 'Vertical', 'Signal', 'XPosition', 
                                      'YPosition', 'Reward', 'ThrustInteraction', 'Radius', 
                                      'Angle', 'Velocity_X', 'Velocity_Y', 'SignalChange', 
                                      'Signal_RollingMean', 'Signal_RollingStd', 'Signal_ThrustInteraction', 
                                      'PHighSignal', 'NHighSignal']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix Heatmap')
        plt.savefig(f"{self.output_dir}/feature_correlation_matrix.png")
        plt.close()
        
    def plot_correlation_matrix(self):
        correlation_matrix = self.df[['Horizontal', 'Vertical', 'Signal', 'XPosition', 'YPosition', 'Reward']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix Heatmap')
        plt.savefig(f"{self.output_dir}/correlation_matrix.png")
        plt.close()
        
        
        
    def game_trajectory_clustering(self):
        # Normalize the position data
        scaled_positions = StandardScaler().fit_transform(self.df[['XPosition', 'YPosition']])
        # Use DBSCAN to identify clusters based on spatial proximity
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(scaled_positions)
        self.df['Cluster'] = clustering.labels_
        # Plotting
        plt.scatter(self.df['XPosition'], self.df['YPosition'], c=self.df['Cluster'])
        plt.title('Trajectory Clustering')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(f"{self.output_dir}/trajectory_clustering.png")
        plt.close()
        

    def trajectory_clustering(self, stop_on_high_reward=True):
        # Normalize the position data
        scaled_positions = StandardScaler().fit_transform(self.df[['XPosition', 'YPosition']])
        # Use DBSCAN to identify clusters based on spatial proximity
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(scaled_positions)
        self.df['Cluster'] = clustering.labels_

        # Calculate the reward threshold
        unique_ids = self.df['AgentID'].unique()
        reward_threshold = (70 / len(unique_ids)) * 0.9

        # Create a color palette with seaborn that has as many colors as there are unique agent IDs
        palette = sns.color_palette("hsv", len(unique_ids))
        color_map = dict(zip(unique_ids, palette))

        # Plotting
        plt.figure(figsize=(10, 8))
        for agent_id in unique_ids:
            agent_data = self.df[self.df['AgentID'] == agent_id]

            if stop_on_high_reward:
                # Identify the first index where the reward exceeds the threshold
                high_reward_index = agent_data[agent_data['Reward'] > reward_threshold].index.min()
                if pd.notna(high_reward_index):
                    # Truncate the data at the first high reward occurrence
                    agent_data = agent_data.loc[:high_reward_index]

            plt.scatter(agent_data['XPosition'], agent_data['YPosition'], 
                        color=color_map[agent_id], label=f'Agent {int(agent_id+1)}', s=50, alpha=0.7)

        plt.title('Trajectory Clustering')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(title='Agent ID')
        plt.savefig(f"{self.output_dir}/trajectory_clustering.png")
        plt.close()

    def agent_density_heatmap(self):
        # Create a grid of the environment
        x_edges = np.linspace(self.df['XPosition'].min(), self.df['XPosition'].max(), num=50)
        y_edges = np.linspace(self.df['YPosition'].min(), self.df['YPosition'].max(), num=50)
        heatmap, _, _ = np.histogram2d(self.df['XPosition'], self.df['YPosition'], bins=[x_edges, y_edges])
        
        # Plotting
        plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Agent Density Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(f"{self.output_dir}/agent_density_heatmap.png")
        plt.close()
  
        
    def apply_pca_and_regression(self, variance_threshold=0.9):
        pca = PCA()
        pca.fit(self.scaled_features)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1 if np.any(cumulative_variance >= variance_threshold) else len(pca.explained_variance_ratio_)

        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(self.scaled_features)
        self.df[[f'PCA_{i+1}' for i in range(n_components)]] = pca_components
        self.df['PCA_Reward'] = self.df['Reward']

        self.plot_pca_variance(pca)
        self.regression_on_pca_components()

    def plot_pca_variance(self, pca):
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('PCA Explained Variance')
        plt.savefig(os.path.join(self.output_dir, "PCA_explained_variance.png"))
        plt.close()

    def regression_on_pca_components(self):
        pca_columns = [col for col in self.df.columns if 'PCA_' in col and 'Reward' not in col]
        X = sm.add_constant(self.df[pca_columns])
        y = self.df['PCA_Reward']
        model = sm.OLS(y, X).fit()
        with open(os.path.join(self.output_dir, 'PCA_regression_summary.txt'), 'w') as f:
            f.write(str(model.summary()))
        

        
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
    
    

    def regression_aggregate(self):
        X = self.df[['Horizontal', 'Vertical', 'Signal', 'ThrustInteraction']]
        y = self.df['Reward']
        X = sm.add_constant(X)  # Adding a constant for the intercept
        model = sm.OLS(y, X).fit()
        self.results['aggregate'] = model.summary()
        # Saving model summary
        with open(f"{self.output_dir}/aggregate_regression_summary.txt", 'w') as file:
            file.write(model.summary().as_text())
            

    def perform_pca_aggregate(self):
        """Performs PCA on the entire dataset."""
        pca = PCA()
        components = pca.fit_transform(self.scaled_features)
        self.save_pca_results(components, 'aggregate', 'aggregate')

    def save_pca_results(self, components, identifier, level):
        """Saves PCA results to files and optionally generates plots."""
        num_components = components.shape[1]
        plt.figure(figsize=(10, 7))
        plt.plot(np.cumsum(PCA().fit(components).explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'PCA Explained Variance - {level.capitalize()} {identifier}')
        plt.savefig(os.path.join(self.output_dir, f'PCA_variance_{level}_{identifier}.png'))
        plt.close()

        
    def analyze_behavioral_impact(self):
        """
        Analyzes the impact of communication on movement by examining correlations.
        """
        # Calculate changes in movement
        self.df['HorizontalChange'] = self.df['Horizontal'].diff().fillna(0)
        self.df['VerticalChange'] = self.df['Vertical'].diff().fillna(0)
        
        # Calculate correlations
        correlation_horizontal = self.df['Signal'].corr(self.df['HorizontalChange'])
        correlation_vertical = self.df['Signal'].corr(self.df['VerticalChange'])
        
        with open(os.path.join(self.output_dir, 'behavioral_correlation.txt'), 'w') as f:
            f.write(f"Correlation between Signal and Horizontal Movement Change: {correlation_horizontal:.3f}\n")
            f.write(f"Correlation between Signal and Vertical Movement Change: {correlation_vertical:.3f}\n")

        return correlation_horizontal, correlation_vertical
    
    
    def path_efficiency_analysis(self):
        # Calculate the direct distance from start to target positions
        start_positions = self.df.groupby('AgentID').first()[['XPosition', 'YPosition']]
        target_positions = self.df.groupby('AgentID').last()[['XPosition', 'YPosition']]
        self.df['StartToEndDist'] = np.sqrt((start_positions['XPosition'] - target_positions['XPosition'])**2 + 
                                            (start_positions['YPosition'] - target_positions['YPosition'])**2)
        # Calculate the actual path distance traveled by summing incremental distances
        self.df['PathDist'] = self.df.groupby('AgentID').apply(lambda x: np.sum(np.sqrt(np.diff(x['XPosition'])**2 + np.diff(x['YPosition'])**2)))

        # Calculate efficiency as the ratio of direct distance to path distance
        self.df['Efficiency'] = self.df['StartToEndDist'] / self.df['PathDist']
        avg_efficiency = self.df['Efficiency'].mean()

        with open(os.path.join(self.output_dir, 'path_efficiency.txt'), 'w') as f:
            f.write(f"Average Path Efficiency: {avg_efficiency:.3f}\n")
        return avg_efficiency



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


    def save_results(self, filename='behavior.txt'):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            
                # Categorizing rewards
                self.df['ThrustInteraction'] = self.df['Reward'].apply(lambda x: 'Food' if x > (69/len(self.unique_agents)) else ('Poison' if x < (-9.9/len(self.unique_agents)) else 'Neutral'))

                # Aggregating results by AgentID and ThrustInteraction
                interaction_counts = self.df.groupby(['AgentID', 'ThrustInteraction']).size().unstack(fill_value=0)
                f.write("ThrustInteraction Counts:\n" + interaction_counts.to_string() + "\n\n")

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

        
    def calculate_individual_agent_entropy(self, signal_column='Signal'):
        """Calculates the entropy for each individual agent's signals."""
        entropies = {}
        for agent_id in self.unique_agents:
            agent_signals = self.df[self.df['AgentID'] == agent_id][signal_column]
            hist, bin_edges = np.histogram(agent_signals, bins=10, density=True)
            probabilities = hist * np.diff(bin_edges)
            entropy_value = entropy(probabilities[probabilities > 0], base=2)
            entropies[agent_id] = entropy_value
            
        return entropies
    
    
    
    
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
        
        est = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='kmeans')
        signal1_discretized = est.fit_transform(signal1.reshape(-1, 1)).flatten()
        signal2_discretized = est.fit_transform(signal2.reshape(-1, 1)).flatten()
        return mutual_info_score(signal1_discretized, signal2_discretized)
    
    def calculate_mutual_info_results(self):
        results = []
        for i in range(len(self.unique_agents)):
            for j in range(i + 1, len(self.unique_agents)):
                agent_i_signals = self.df[self.df['AgentID'] == self.unique_agents[i]]['Signal']
                agent_j_signals = self.df[self.df['AgentID'] == self.unique_agents[j]]['Signal']
                
                if len(agent_i_signals) > 0 and len(agent_j_signals) > 0:
                    mi = self.calculate_mutual_information(agent_i_signals.values, agent_j_signals.values)
                    results.append({'agents': (self.unique_agents[i], self.unique_agents[j]), 'MI': mi})
                    
        self.mutual_info_results = results
        return results

    

               
     
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
            entropy_value = self.calculate_entropy(agent_data['Signal'])
            correlation = agent_data['Signal'].corr(agent_data['Reward'])
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
        pca = PCA()
        pca_result = pca.fit_transform(self.scaled_features)
        n_components = min(2, pca_result.shape[1])  # Limit to 2 components
        self.df[[f'PCA{i+1}' for i in range(n_components)]] = pca_result[:, :n_components]
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
        signal = self.df['Signal'].values
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
        pca = PCA().fit(self.scaled_features)
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'cumulative_variance.png'))
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
        
        signal = self.df['Signal'].values
        f, t, Sxx = spectrogram(signal, fs=1)  # Assuming 1 Hz sampling rate; adjust as necessary
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')
        plt.title('Spectrogram of Communication Signal')
        plt.colorbar(label='Intensity (dB)')
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()
        
    

        
    

        
        
    
    
    

        
        





