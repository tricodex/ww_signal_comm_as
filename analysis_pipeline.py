import warnings
# Filter out TensorFlow oneDNN warning:
warnings.filterwarnings("ignore", message="oneDNN custom operations are on*")

# Filter out Keras deprecation warnings:
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Filter out Scikit-learn's small bins warnings:
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._discretization")

# Filter out Scikit-learn's K-means FutureWarning 
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

# Filter out UserWarning for set_ticklabels()
warnings.filterwarnings("ignore", category=UserWarning, message="set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.")


# analysis_pipeline.py
import re
import os
import datetime
import pickle
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from heurisitic_signal_policy import communication_heuristic_policy
import waterworld_v4 
import seaborn as sns
from analysis import Analysis
from combine import combine_reports
import matplotlib.pyplot as plt

env_fn = waterworld_v4
current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


model_test_configs = [
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240313-111355.zip", "n_pursuers": 2},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-011623.zip", "n_pursuers": 4},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.04, "poison_speed": 0.15},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240405-125529.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
    {"model_name": "SAC", "model_path": "models/train/waterworld_v4_20240315-171531.zip", "n_pursuers": 2},
    {"model_name": "SAC", "model_path": "models/train/waterworld_v4_20240318-022243.zip", "n_pursuers": 4},
    {"model_name": "SAC", "model_path": "models/train/waterworld_v4_20240314-195426.zip", "n_pursuers": 6},
    {"model_name": "SAC", "model_path": "models/train/waterworld_v4_20240406-122632.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
    {"model_name": "Heuristic", "model_path": None, "n_pursuers": 4},
    {"model_name": "Heuristic", "model_path": None, "n_pursuers": 6},
    {"model_name": "Heuristic", "model_path": None, "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
]

sensor_test_configs = [
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.02, "poison_speed": 0.075},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.04, "poison_speed": 0.075},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.1, "poison_speed": 0.075},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.4, "poison_speed": 0.075},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.8, "poison_speed": 0.075},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.2, "poison_speed": 0.075},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.2, "poison_speed": 0.15},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240405-125529.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240405-125529.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.075, "sensor_count": 8},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240405-125529.zip", "n_pursuers": 8, "sensor_range": 0.2, "poison_speed": 0.15, "sensor_count": 8},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240405-125529.zip", "n_pursuers": 8, "sensor_range": 0.2, "poison_speed": 0.075, "sensor_count": 8},
]



def eval_with_model_path_run(env_fn, model_path, model_name, num_pursuers, sensor_range=None, poison_speed=None, sensor_count=None, num_games=100, render_mode=None):
    env_kwargs = {
        "n_pursuers": num_pursuers,
        "n_evaders": 6,
        "n_poisons": 8,
        "n_coop": 2,
        "n_sensors": sensor_count if sensor_count is not None else 16,
        "sensor_range": sensor_range if sensor_range is not None else 0.2,
        "radius": 0.015,
        "obstacle_radius": 0.055,
        "n_obstacles": 1,
        "obstacle_coord": [(0.5, 0.5)],
        "pursuer_max_accel": 0.01,
        "evader_speed": 0.01,
        "poison_speed": poison_speed if poison_speed is not None else 0.075,
        "poison_reward": -10,
        "food_reward": 70.0,
        "encounter_reward": 0.015,
        "thrust_penalty": -0.01,
        "local_ratio": 0.0,
        "speed_features": True,
        "max_cycles": 1000
    }
    actions = []
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    if model_path is None:
        if model_name == "Heuristic":
            print(f"Proceeding with heuristic policy for {num_games} games.")
        else:
            print("Model path is None but model name is not 'Heuristic'.")
            return None
    else:
        if not os.path.exists(model_path):
            print("Model not found.")
            return None
        if model_name == "PPO":
            model = PPO.load(model_path)
        elif model_name == "SAC":
            model = SAC.load(model_path)
        else:
            print("Invalid model name.")
            return None

    total_rewards = {agent: 0 for agent in env.possible_agents}
    episode_avg_rewards = []

    for i in range(num_games):
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            episode_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                if model_name == "Heuristic":
                    n_sensors = env_kwargs.get('n_sensors')
                    action = communication_heuristic_policy(obs, n_sensors, env_kwargs['sensor_range'], len(env.possible_agents))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    
                
                
                # Retrieve individual position from the info dictionary
                position = info.get('pursuer_position', np.array([0, 0]))  # Default to [0, 0] if no data available

                # Store action, individual position, reward, and agent ID as a single array for later analysis
                actionid = np.concatenate((action, position, [reward, int(agent[-1])]))  # Ensure this matches the expected input format for Analysis
                actions.append(actionid)
                
            env.step(action)

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg_rewards.append(sum(episode_rewards.values()) / len(episode_rewards))

    env.close()

    overall_avg_reward = sum(total_rewards.values()) / (len(total_rewards) * num_games)
    print("Total Rewards: ", total_rewards, "over", num_games, "games")
    print(f"Overall Avg reward: {overall_avg_reward}")
    
    data = {'actions': actions, 'overall_avg_reward': overall_avg_reward, 
            'config_details': model_test_configs, 
            'episode_avg_rewards': episode_avg_rewards}
    
    file_save = f'pickles/{current_datetime}/{model_name}_{num_pursuers}_s{sensor_range}eval_results_{num_games}.pkl'
    os.makedirs(os.path.dirname(file_save), exist_ok=True)
    
    # save data to pickle
    with open(file_save, 'wb') as file:
        pickle.dump(data, file)

    return data



def parse_filename(filename):
    match = re.match(r'(\w+)_(\d+)_s(.*?)eval_results_\d+.pkl', filename)
    if match:
        model_name, n_pursuers, sensor_range = match.groups()
        return {
            'model_name': model_name,
            'n_pursuers': int(n_pursuers),
            'sensor_range': None if sensor_range == 'None' else float(sensor_range)
        }
    return None

def find_config(parsed_filename, configs):
    for config in configs:
        if (config['model_name'] == parsed_filename['model_name'] and
            config['n_pursuers'] == parsed_filename['n_pursuers'] and
            config.get('sensor_range', None) == parsed_filename['sensor_range']):
            return config
    return None


def run_and_analyze_all_configs_from_pickle(directory=r'C:\Users\patri\Desktop\thesis\code\ww_signal_comm_as\pickles\20240430-023536', configs=model_test_configs, games=100):
    all_results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            parsed_filename = parse_filename(filename)
            if parsed_filename:
                config = find_config(parsed_filename, configs)
                if config:
                    with open(filepath, 'rb') as file:
                        eval_results = pickle.load(file)
                    
                    # Proceed with analysis assuming 'Analysis' is defined and configured
                    data = eval_results.get('actions', [])
                    avg_reward = eval_results.get('overall_avg_reward', 'Data Not Available')
                    
                    if config.get('sensor_range') is not None:
                        config_key = f"{config['model_name']}_pursuers_{config['n_pursuers']}M"
                    else:
                        config_key = f"{config['model_name']}_pursuers_{config['n_pursuers']}"
                    output_dir = f"results/{current_datetime}/{config_key}"
            
                    analysis = Analysis(data, output_dir=output_dir, episodes=evals)
                    print("Plotting feature correlation matrix...")
                    analysis.plot_feature_correlation_matrix()
                    print("Performing feature analysis...")
                    #feature_analysis()
                    print("Analyzing rewards correlation...")
                    analysis.analyze_rewards_correlation()
                    print("Clustering by reward category...")
                    analysis.cluster_by_reward_category()
                    print("Analyzing dynamic behavior before encounters...")
                    analysis.dynamic_behavior_before_encounters()
                    # print("Plotting time series analysis...")
                    # analysis.plot_time_series_analysis()
                    
                    #full_analysis()
                    print("Performing individual analysis...")
                    analysis.individual_analysis()
                    print("Performing collective analysis...")
                    analysis.collective_analysis()
                    print("Analyzing across evaluations...")
                    analysis.analysis_across_evaluations()
                    print("Saving analysis results...")
                    analysis.save_analysis_results()
                    print(f"Analysis results for {config['model_name']} with {config['n_pursuers']} pursuers: {analysis.results['evaluation']}")

                    all_results[config_key] = {
                                                    'data': data,
                                                    'analysis_results': analysis.results,
                                                    'avg_reward': avg_reward
                                                }
                    
                    
                    
                    print("Applying PCA and regression...")
                    analysis.apply_pca_and_regression()
                    
                    print("Performing PCA aggregate...")
                    analysis.perform_pca_aggregate()
                    
                    
                    print("Analyzing signal behavior before encounters...")
                    analysis.signal_behavior_before_encounters()
                    # print("Performing comprehensive analysis...")
                    # analysis.perform_comprehensive_analysis()
                    
                    
                    # print("Performing behavior clustering... not...")
                    
                    print("Performing trajectory clustering... not...")
                    #analysis.trajectory_clustering()
                    
                else:
                    print(f"Failed to get results for {config['model_name']} with {config['n_pursuers']} pursuers.")

    combine_dir = f"results/{current_datetime}"
    combine_reports(combine_dir)
    
    
    with open('pickles/all_results.pkl', 'wb') as file:
                pickle.dump(all_results, file)
            
            
            
    return all_results
        
def run_and_analyze_all_configs(games=100):
    all_results = {}  

    for config in model_test_configs:
        config_suffix = "M" if "sensor_range" in config else ""
        print(" ---",
              "\n\n",
              f"Evaluating configuration: {config['model_name']} with {config['n_pursuers']} pursuers",
              "\n\n",
              "---"
              
              )
        


        # Run the model evaluation
        eval_results = eval_with_model_path_run(
            env_fn=env_fn,
            model_path=config.get("model_path"),
            model_name=config["model_name"],
            num_pursuers=config["n_pursuers"],
            sensor_range=config.get("sensor_range"),
            poison_speed=config.get("poison_speed"),
            sensor_count=config.get("sensor_count"),
            num_games=games,
            render_mode=None
        )
        

        if eval_results:
            data = eval_results.get('actions', [])
            avg_reward = eval_results.get('overall_avg_reward', 'Data Not Available')

            
            config_key = f"{config['model_name']}_pursuers_{config['n_pursuers']}{config_suffix}"
            output_dir = f"results/{current_datetime}/{config_key}"
            
            analysis = Analysis(data, output_dir=output_dir, episodes=evals)
            print("Plotting feature correlation matrix...")
            analysis.plot_feature_correlation_matrix()
            print("Performing feature analysis...")
            #feature_analysis()
            print("Analyzing rewards correlation...")
            analysis.analyze_rewards_correlation()
            print("Clustering by reward category...")
            analysis.cluster_by_reward_category()
            print("Analyzing dynamic behavior before encounters...")
            analysis.dynamic_behavior_before_encounters()
            print("Plotting time series analysis...")
            analysis.plot_time_series_analysis()
            
            #full_analysis()
            print("Performing individual analysis...")
            analysis.individual_analysis()
            print("Performing collective analysis...")
            analysis.collective_analysis()
            print("Analyzing across evaluations...")
            analysis.analysis_across_evaluations()
            print("Saving analysis results...")
            analysis.save_analysis_results()
            print(f"Analysis results for {config['model_name']} with {config['n_pursuers']} pursuers: {analysis.results['evaluation']}")

            all_results[config_key] = {
                                            'data': data,
                                            'analysis_results': analysis.results,
                                            'avg_reward': avg_reward
                                        }
            
            
            
            print("Applying PCA and regression...")
            analysis.apply_pca_and_regression()
            
            print("Performing PCA aggregate...")
            analysis.perform_pca_aggregate()
            
            
            print("Analyzing signal behavior before encounters...")
            analysis.signal_behavior_before_encounters()
            # print("Performing comprehensive analysis...")
            # analysis.perform_comprehensive_analysis()
            
            
            # print("Performing behavior clustering... not...")
            
            print("Performing trajectory clustering...")
            analysis.trajectory_clustering()
            # print("Generating agent density heatmap...")
            # analysis.agent_density_heatmap()
            # print("Performing spectral clustering...")
            # analysis.cluster(method='spectral')
            
            # print("Plotting clusters...")
            # analysis.plot_clusters()
            # print("Performing time frequency analysis... not...")
            #analysis.perform_time_frequency_analysis(plot_name=f"time_freq_analysis.png")
                    
            
        else:
            print(f"Failed to get results for {config['model_name']} with {config['n_pursuers']} pursuers.")

    combine_dir = f"results/{current_datetime}"
    combine_reports(combine_dir)
    
    
    with open('pickles/all_results.pkl', 'wb') as file:
                pickle.dump(all_results, file)
    
    
    
    return all_results



def compare_across_configurations(all_results):
    print("Comparing configurations across all results.")
    config_data = {
        'Configuration': [],
        'Average Reward': [],
        'Average Entropy': [],
        'Average Mutual Information': []
    }

    for config, results in all_results.items():
        config_data['Configuration'].append(config)
        
        # Accessing the evaluation data correctly
        evaluation_data = results.get('analysis_results', {}).get('evaluation', {})
        avg_reward = evaluation_data.get('average_reward')
        avg_entropy = evaluation_data.get('average_entropy')
        
        # Handling the list of dictionaries for mutual information
        avg_mutual_info_list = evaluation_data.get('average_mutual_information', [])
        if avg_mutual_info_list and isinstance(avg_mutual_info_list, list):
            avg_mutual_information = avg_mutual_info_list[0].get('MI') if avg_mutual_info_list else None
        else:
            avg_mutual_information = None

        config_data['Average Reward'].append(avg_reward if avg_reward is not None else 'No data')
        config_data['Average Entropy'].append(avg_entropy if avg_entropy is not None else 'No data')
        config_data['Average Mutual Information'].append(avg_mutual_information if avg_mutual_information is not None else 'No data')

    df = pd.DataFrame(config_data)
    df['Configuration'] = df['Configuration'].astype('category')
    df['Average Reward'] = pd.to_numeric(df['Average Reward'], errors='coerce')
    df['Average Mutual Information'] = pd.to_numeric(df['Average Mutual Information'], errors='coerce')

    
    
    
    return df




def rune_gamelevel_analysis(config):
    # Configuration details for directory and file naming
    sensor_range_str = str(config["sensor_range"]).replace('.', 'p')  # Replace dot with 'p' for file system compatibility
    poison_speed_str = str(config["poison_speed"]).replace('.', 'p')
    config_suffix = f"sr{sensor_range_str}_ps{poison_speed_str}"

    output_dir = f"results/{current_datetime}_gamelevel/{config['model_name']}_pursuers_{config['n_pursuers']}_{config_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning advanced analysis for {config['model_name']} with {config['n_pursuers']} agents, Sensor Range: {config['sensor_range']}, Poison Speed: {config['poison_speed']}\n")
    eval_results = eval_with_model_path_run(
        env_fn=env_fn,
        model_path=config.get("model_path"),
        model_name=config["model_name"],
        num_pursuers=config["n_pursuers"],
        sensor_range=config.get("sensor_range"),
        poison_speed=config.get("poison_speed"),
        sensor_count=config.get("sensor_count"),
        num_games=evals,  # Focused analysis on a single, detailed game
        render_mode=None
    )
    
    if eval_results:
        actions = eval_results.get('actions', [])
        analysis = Analysis(actions, output_dir=output_dir)

        # Perform and save each analysis with filenames reflecting the configuration details
        analysis.apply_hierarchical_clustering()
        analysis.plot_hierarchical_clusters(n_clusters=5, plot_name=f"hierarchical_clusters.png")
        analysis.perform_time_frequency_analysis(plot_name=f"time_freq_analysis.png")
        analysis.plot_signal_histogram(plot_name=f"signal_histogram.png")
        analysis.create_k_distance_plot(plot_name=f"k_distance_plot.png")
        
        
        
        analysis.signal_behavior_before_encounters()
        # analysis.perform_comprehensive_analysis()
        
        
        analysis.trajectory_clustering()
        analysis.agent_density_heatmap()
        analysis.cluster(method='spectral')
        analysis.cluster(method='kmeans')
        analysis.plot_clusters()
        


        print(f"Completed advanced analysis for {config['model_name']} with {config['n_pursuers']} agents.")
    else:
        print("Failed to obtain results for advanced analysis.")
        
def create_comparison_plot():        
    # Convert the 'Configuration' column to a category type for proper plotting
    df = pd.read_csv(f'results/{current_datetime}/comparative_analysis.csv')
    df['Configuration'] = df['Configuration'].astype('category')

    # Generate the bar plots
    fig, ax = plt.subplots(3, figsize=(10, 15), dpi=100)
    sns.barplot(x='Configuration', y='Average Reward', hue='Configuration', data=df, ax=ax[0], palette='Blues', legend=False)
    sns.barplot(x='Configuration', y='Average Entropy', hue='Configuration', data=df, ax=ax[1], palette='Greens', legend=False)
    sns.barplot(x='Configuration', y='Average Mutual Information', hue='Configuration', data=df, ax=ax[2], palette='Oranges', legend=False)

    # Adjust the x-axis labels to center them under each bar
    for axis in ax:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha='right')

    # Adjust figure size and spacing
    fig.set_size_inches(10, 18)  # Increase the height of the figure
    fig.subplots_adjust(hspace=0.5)  # Increase the vertical spacing between subplots

    # Show the plot
    plt.savefig(f"results/{current_datetime}/comparison_plot.png")
    
    # Summarizing key insights
    key_findings = {
        'Best Performing Model': df.loc[df['Average Reward'].idxmax()]['Configuration'],
        'Highest MI': df.loc[df['Average Mutual Information'].idxmax()]['Configuration'],
        'Lowest Entropy': df.loc[df['Average Entropy'].idxmin()]['Configuration'],
    }
    with open(f"results/{current_datetime}/key_findings.txt", 'w') as file:
        file.write(str(key_findings))
    print("Key Findings:", key_findings)


if __name__ == "__main__":
    evals = 1
    process_to_run = 'analysis'
    output_dir = f"results/{current_datetime}_{evals}evals"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if process_to_run == 'analysis':
        all_results = run_and_analyze_all_configs(games=evals)
        comparative_data = compare_across_configurations(all_results)
        comparative_data.to_csv(f'results/{current_datetime}/comparative_analysis.csv', index=False)
        create_comparison_plot()
        
    elif process_to_run == 'gamalevel_analysis':
        for config in sensor_test_configs:
            rune_gamelevel_analysis(config)
            
    elif process_to_run == 'eval':
        eval_results = eval_with_model_path_run(
            env_fn=env_fn,
            model_path="models/train/waterworld_v4_20240314-050633.zip",
            model_name="PPO",
            num_pursuers=6,
            sensor_range=0.2,
            poison_speed=0.075,
            sensor_count=16,
            num_games=evals,
            render_mode=None
        )
        
    elif process_to_run == 'pickle':
        all_results = run_and_analyze_all_configs_from_pickle()
        comparative_data = compare_across_configurations(all_results)
        comparative_data.to_csv(f'results/{current_datetime}/comparative_analysis.csv', index=False)
        create_comparison_plot()
        
        
test_test_configs = [
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240314-050633.zip", "n_pursuers": 6, "sensor_range": 0.04, "poison_speed": 0.15},
    {"model_name": "PPO", "model_path": "models/train/waterworld_v4_20240405-125529.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
]