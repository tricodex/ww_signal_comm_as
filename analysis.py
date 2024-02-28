import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.signal import welch
from scipy.stats import entropy
import statsmodels.api as sm


class Analysis:
    def __init__(self, actions_array):
        self.actions_array = actions_array
        self.df = pd.DataFrame(actions_array, columns=['Horizontal', 'Vertical', 'Communication', 'AgentID'])
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[['Horizontal', 'Vertical', 'Communication']])
        self.pca = PCA(n_components=2)
        self.pca_components = self.pca.fit_transform(self.scaled_features)
        self.pca_df = pd.DataFrame(data=self.pca_components, columns=['PCA1', 'PCA2'])
        self.pca_df['AgentID'] = self.df['AgentID']
        self.X = sm.add_constant(self.df['Communication'])
        self.y = self.df['Horizontal']
        self.model = sm.OLS(self.y, self.X).fit()
        self.df['Predicted_Horizontal'] = self.model.predict(self.X)
        self.residuals = self.model.resid
        self.kmeans = KMeans(n_clusters=4, random_state=0)
        self.df['Cluster'] = self.kmeans.fit_predict(self.scaled_features)
        self.regression_model = LinearRegression().fit(self.df[['Communication']], self.df['Horizontal'])
        self.df['Predicted_Horizontal'] = self.regression_model.predict(self.df[['Communication']])
        self.unique_agents = self.df['AgentID'].unique()
        self.mutual_info_results = []

    def calculate_mutual_information(self, signal1, signal2, n_bins=10):
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        signal1_discretized = est.fit_transform(signal1.reshape(-1, 1)).flatten()
        signal2_discretized = est.fit_transform(signal2.reshape(-1, 1)).flatten()
        return mutual_info_score(signal1_discretized, signal2_discretized)

    def calculate_mutual_info_results(self):
        for i in range(len(self.unique_agents)):
            for j in range(i + 1, len(self.unique_agents)):
                agent_i_signals = self.df[self.df['AgentID'] == self.unique_agents[i]]['Communication'].values
                agent_j_signals = self.df[self.df['AgentID'] == self.unique_agents[j]]['Communication'].values
                mi = self.calculate_mutual_information(agent_i_signals, agent_j_signals)
                self.mutual_info_results.append((self.unique_agents[i], self.unique_agents[j], mi))

    def print_mutual_info_results(self):
        for result in self.mutual_info_results:
            print(f'Mutual information between Agent {result[0]} and Agent {result[1]}: {result[2]}')

    def save_mutual_info_results(self, filepath='mutual_information_results.txt'):
        with open(filepath, 'w') as f:
            for result in self.mutual_info_results:
                f.write(f'Mutual information between Agent {result[0]} and Agent {result[1]}: {result[2]}\n')

    def plot_movement_scatter(self, plot_name='movement_scatter_plot.png'):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['AgentID'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Agent ID')
        plt.title('Movement Scatter Plot Color-coded by Agent ID')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        plt.grid(True)
        plt.savefig(f'plots/hvi/{plot_name}')
        plt.show()

    def plot_movement_communication_scatter(self, plot_name='movement_communication_scatter_plot.png'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.df['Horizontal'], self.df['Vertical'], self.df['Communication'], c=self.df['AgentID'], cmap='viridis', alpha=0.5)
        fig.colorbar(scatter, ax=ax, label='Agent ID')
        ax.set_title('3D Scatter Plot of Movements and Communication Signal, Color-coded by Agent ID')
        ax.set_xlabel('Horizontal Movement')
        ax.set_ylabel('Vertical Movement')
        ax.set_zlabel('Communication Signal')
        plt.savefig(f'plots/hvsi/{plot_name}')
        plt.show()

    def plot_pca_results(self, plot_name='pca_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.pca_df['PCA1'], self.pca_df['PCA2'], c=self.pca_df['AgentID'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Agent ID')
        plt.title('PCA: 2 Principal Components of Action Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(f'plots/analysis/{plot_name}')
        plt.show()

    def plot_clustering_results(self, plot_name='clustering_plot.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['Horizontal'], self.df['Vertical'], c=self.df['Cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Clustering: Agent Behavior Modeling')
        plt.xlabel('Horizontal Movement')
        plt.ylabel('Vertical Movement')
        plt.savefig(f'plots/analysis/{plot_name}')
        plt.show()

    def plot_residuals_vs_predicted(self, plot_name='residuals_vs_predicted_plot.png'):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Predicted_Horizontal'], self.residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Horizontal Movement')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Horizontal Movement')
        plt.savefig(f'plots/analysis/{plot_name}')
        plt.show()

    def plot_residuals_histogram(self, plot_name='residuals_histogram.png'):
        plt.figure(figsize=(10, 6))
        plt.hist(self.residuals, bins=20, edgecolor='k')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.savefig(f'plots/analysis/{plot_name}')
        plt.show()

    def plot_residuals_qq_plot(self, plot_name='residuals_qq_plot.png'):
        fig = sm.qqplot(self.residuals, line='45')
        plt.title('Q-Q Plot of Residuals')
        plt.savefig(f'plots/analysis/{plot_name}')
        plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mutual_info_score
# from sklearn.preprocessing import KBinsDiscretizer
# from scipy.signal import welch
# from scipy.stats import entropy
# import statsmodels.api as sm


# # Convert to DataFrame for easier manipulation
# df = pd.DataFrame(actions_array, columns=['Horizontal', 'Vertical', 'Communication', 'AgentID'])

# # Initialize a StandardScaler instance
# scaler = StandardScaler()

# # Standardizing the features for PCA and clustering
# scaled_features = scaler.fit_transform(df[['Horizontal', 'Vertical', 'Communication']])

# # PCA transformation
# pca = PCA(n_components=2)
# pca_components = pca.fit_transform(scaled_features)
# pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
# pca_df['AgentID'] = df['AgentID']

# # Linear Regression to predict Horizontal Movement from Communication Signal
# X = sm.add_constant(df['Communication'])  # Adds a constant term to the predictor
# y = df['Horizontal']

# # Fit the regression model using statsmodels for detailed statistics
# model = sm.OLS(y, X).fit()
# df['Predicted_Horizontal'] = model.predict(X)

# # Calculate residuals for further analysis
# residuals = model.resid

# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=4, random_state=0)
# df['Cluster'] = kmeans.fit_predict(scaled_features)

# # Linear Regression: Predicting Horizontal Movement based on Communication Signal
# regression_model = LinearRegression().fit(df[['Communication']], df['Horizontal'])
# df['Predicted_Horizontal'] = regression_model.predict(df[['Communication']])

# # For each unique pair of agents, calculate mutual information
# unique_agents = df['AgentID'].unique()
# mutual_info_results = []

# def calculate_mutual_information(signal1, signal2, n_bins=10):
#     est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
#     signal1_discretized = est.fit_transform(signal1.reshape(-1, 1)).flatten()
#     signal2_discretized = est.fit_transform(signal2.reshape(-1, 1)).flatten()
#     return mutual_info_score(signal1_discretized, signal2_discretized)

# # Iterating over each unique pair of agents to calculate mutual information
# for i in range(len(unique_agents)):
#     for j in range(i + 1, len(unique_agents)):
#         agent_i_signals = df[df['AgentID'] == unique_agents[i]]['Communication'].values
#         agent_j_signals = df[df['AgentID'] == unique_agents[j]]['Communication'].values
#         mi = calculate_mutual_information(agent_i_signals, agent_j_signals)
#         mutual_info_results.append((unique_agents[i], unique_agents[j], mi))

# # Print or save the mutual information results
# for result in mutual_info_results:
#     print(f'Mutual information between Agent {result[0]} and Agent {result[1]}: {result[2]}')

# # Save mutual information results to a file
# with open('mutual_information_results.txt', 'w') as f:
#     for result in mutual_info_results:
#         f.write(f'Mutual information between Agent {result[0]} and Agent {result[1]}: {result[2]}\n')

# # Plotting the Movement Scatter Plot Color-coded by Agent ID
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(df['Horizontal'], df['Vertical'], c=df['AgentID'], cmap='viridis', alpha=0.5)
# plt.colorbar(scatter, label='Agent ID')
# plt.title('Movement Scatter Plot Color-coded by Agent ID')
# plt.xlabel('Horizontal Movement')
# plt.ylabel('Vertical Movement')
# plt.grid(True)
# plot_name = 'movement_scatter_plot.png'
# plt.savefig(f'plots/hvi/{plot_name}')
# plt.show()

# # Plotting the 3D Scatter Plot incorporating Communication Signal
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(df['Horizontal'], df['Vertical'], df['Communication'], c=df['AgentID'], cmap='viridis', alpha=0.5)
# fig.colorbar(scatter, ax=ax, label='Agent ID')
# ax.set_title('3D Scatter Plot of Movements and Communication Signal, Color-coded by Agent ID')
# ax.set_xlabel('Horizontal Movement')
# ax.set_ylabel('Vertical Movement')
# ax.set_zlabel('Communication Signal')
# plot_name = 'movement_communication_scatter_plot.png'
# plt.savefig(f'plots/hvsi/{plot_name}')
# plt.show()

# # Plotting PCA Results
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['AgentID'], cmap='viridis', alpha=0.5)
# plt.colorbar(scatter, label='Agent ID')
# plt.title('PCA: 2 Principal Components of Action Space')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plot_name = 'pca_plot.png'
# plt.savefig(f'plots/analysis/{plot_name}')
# plt.show()

# # Plotting Clustering Results
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(df['Horizontal'], df['Vertical'], c=df['Cluster'], cmap='viridis', alpha=0.5)
# plt.colorbar(scatter, label='Cluster')
# plt.title('Clustering: Agent Behavior Modeling')
# plt.xlabel('Horizontal Movement')
# plt.ylabel('Vertical Movement')
# plot_name = 'clustering_plot.png'
# plt.savefig(f'plots/analysis/{plot_name}')
# plt.show()

# # Plotting the Residuals vs Predicted Values for the Linear Regression Model
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Predicted_Horizontal'], residuals, alpha=0.5)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('Predicted Horizontal Movement')
# plt.ylabel('Residuals')
# plt.title('Residuals vs Predicted Horizontal Movement')
# plot_name = 'residuals_vs_predicted_plot.png'
# plt.savefig(f'plots/analysis/{plot_name}')
# plt.show()

# # Plotting the Histogram of Residuals
# plt.figure(figsize=(10, 6))
# plt.hist(residuals, bins=20, edgecolor='k')
# plt.xlabel('Residuals')
# plt.ylabel('Frequency')
# plt.title('Histogram of Residuals')
# plot_name = 'residuals_histogram.png'
# plt.savefig(f'plots/analysis/{plot_name}')
# plt.show()

# # Plotting the Q-Q Plot for Residuals
# fig = sm.qqplot(residuals, line='45')
# plt.title('Q-Q Plot of Residuals')
# plot_name = 'residuals_qq_plot.png'
# plt.savefig(f'plots/analysis/{plot_name}')
# plt.show()

