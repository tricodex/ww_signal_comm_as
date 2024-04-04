# utils.py

import os 
import logging
import datetime
import glob
import numpy as np
from analysis import Analysis
  
MODEL_DIR = os.path.join('models', 'train')

EXPERIMENT_BASE_DIR = 'experiments'

architecture = 128

global_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def find_latest_model(model_dir, model_name_prefix, model_subdir="train"):
    # Adjust the pattern to search within the nested 'train' directories as well
    pattern = os.path.join(model_dir, model_subdir, "*", model_name_prefix + '*.zip')
    list_of_files = glob.glob(pattern, recursive=True)  # Ensure recursive search if necessary
    
    print(f"Searching for models with pattern: {pattern}")
    print(f"Found files: {list_of_files}")
    for model_file in list_of_files:
        print(f"Found model: {model_file}")

    
    if not list_of_files:
        print("No models found with the given pattern.")
        return None
    latest_model = max(list_of_files, key=os.path.getctime)
    return latest_model


def setup_logging(env_name, n_pursuers, model_name, mode):
    log_dir = f"logs/{mode}"
    #timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_filename = f"{mode}_{env_name}_{global_timestamp}_a{n_pursuers}_{model_name}.txt"
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logging
    logging.basicConfig(filename=os.path.join(log_dir, log_filename), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif callable(obj):  # Checks if the object is callable (like functions, lambdas)
        return repr(obj)  # Return string representation
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj
    
# log_experiment_results 
def log_experiment_results(config, score, experiment_log_dir):
    log_path = os.path.join(experiment_log_dir, "experiment_results.log")
    with open(log_path, 'a') as log_file:  # Ensure we're appending to the file
        log_file.write(f"{config['algorithm']} with {config['agents']} agents {'with' if config['communication'] else 'without'} communication: Score = {score}\n")


def setup_experiment_logging():
    
    experiment_log_dir = os.path.join(EXPERIMENT_BASE_DIR, f"experiment_{global_timestamp}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    
    # Setup a root logger or other logging configurations here, if necessary
    log_filename = os.path.join(experiment_log_dir, 'experiment_logs.txt')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    logging.info("Experiment started")
    return experiment_log_dir

def get_configuration(model_name):
    #common_settings = {'env_kwargs': env_kwargs, 'architecture': 128}
    if model_name == 'PPO':
        hyperparams = {
            'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),
            'n_steps': 500,
            'batch_size': 128,
            'n_epochs': 10,
            'gamma': 0.998,
            'gae_lambda': 0.92,
            'clip_range': lambda epoch: 0.1 + 0.15 / (1.0 + 0.1 * epoch),
            'ent_coef': 0.005,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': True,
            'sde_sample_freq': 64,
            'normalize_advantage': True,
        }
    elif model_name == 'SAC':
        hyperparams = {
            'learning_rate': 0.0003,
            'buffer_size': 1000000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'ent_coef': 'auto',
            'target_entropy': 'auto',
            'use_sde': False,
            'sde_sample_freq': -1,
        }
    elif model_name == 'Heuristic':
        hyperparams = {}  # No hyperparameters for heuristic models
    else:
        raise ValueError("Invalid model name. Please choose either 'PPO' or 'SAC'.")

    return hyperparams


def generate_configurations(n_pursuers_options, model_names, include_no_comm=False):
    configurations = []
    for n_pursuers in n_pursuers_options:
        for model_name in model_names:
            configurations.append({
                "algorithm": model_name,
                "agents": n_pursuers,
                "communication": True
            })
            if include_no_comm:
                configurations.append({
                    "algorithm": model_name,
                    "agents": n_pursuers,
                    "communication": False
                })
    return configurations

def print_evaluation_results(total_rewards, num_games):
    avg_rewards = {agent: reward / num_games for agent, reward in total_rewards.items()}
    print(f"Average rewards after {num_games} games:", avg_rewards)


def perform_detailed_analysis(actions):
    base_dir = 'analysis_data'
    os.makedirs(base_dir, exist_ok=True)

    analysis_output_dir = os.path.join(base_dir, global_timestamp)
    os.makedirs(analysis_output_dir, exist_ok=True) 

    # Assuming 'actions' is the dataset
    actions_array = np.array(actions)  # Convert the dataset to a numpy array

    # Instantiate the Analysis class with the actions array and the output directory
    analysis = Analysis(actions_array, analysis_output_dir)  

    # Perform mutual information calculation between all pairs of agents
    analysis.calculate_mutual_info_results()

    # Print mutual information results to console
    analysis.print_mutual_info_results()

    analysis.save_mutual_info_results(filename='mutual_information_results.txt')

    # Apply DBSCAN and Hierarchical clustering
    analysis.apply_dbscan(eps=0.5, min_samples=5)
    analysis.apply_hierarchical_clustering(method='ward')

    # Plot various analyses and visualizations
    analysis.plot_movement_scatter(plot_name='movement_scatter_plot.png')
    analysis.plot_communication_over_time(plot_name='communication_over_time.png')
    analysis.calculate_correlation_with_performance()
    analysis.plot_pca_results(plot_name='pca_plot.png')
    analysis.plot_clustering_results(plot_name='clustering_plot.png')
    analysis.plot_residuals_vs_predicted(plot_name='residuals_vs_predicted_plot.png')
    
    analysis.plot_dbscan_results(plot_name='dbscan_clustering_plot.png')
    
    analysis.save_analysis_results(filename='analysis_results.txt')
    analysis.perform_time_frequency_analysis(plot_name='psd_plot.png')
    
    analysis.plot_autocorrelation()