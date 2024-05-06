# main.py

from __future__ import annotations
import glob
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import supersuit as ss
from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
import pandas as pd
import seaborn as sns

import pickle



#from pettingzoo.sisl import waterworld_v4
import waterworld_v4 
from ga import GeneticHyperparamOptimizer
from settings import env_kwargs
import datetime
current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
evals = 1000
current_datetime = current_datetime + f"_{evals}evals"  

from heurisitic_signal_policy import communication_heuristic_policy
import numpy as np
from analysis import Analysis
import warnings
from combine import combine_reports


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



MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'
architecture = 128

def train_waterworld(env_fn, model_name, model_subdir, steps=100_000, seed=None, **hyperparam_kwargs):
    
    if 'n_steps' in hyperparam_kwargs:
        hyperparam_kwargs['n_steps'] = int(hyperparam_kwargs['n_steps'])
    if 'batch_size' in hyperparam_kwargs:
        hyperparam_kwargs['batch_size'] = int(hyperparam_kwargs['batch_size'])
    if 'buffer_size' in hyperparam_kwargs:
        hyperparam_kwargs['buffer_size'] = int(hyperparam_kwargs['buffer_size'])
    
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    
    # Create the Tensorboard callback
    # log_dir = os.path.join("tensorboard_logs", f"{env.unwrapped.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}") 
    #tensorboard_callback = TensorboardCallback(log_dir=log_dir)

       
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 16, num_cpus=4, base_class="stable_baselines3") # [1]=8
    
    
    # policy_kwargs = dict(net_arch=[128, 64]) :D 8 ,8, 0.04setup
    policy_kwargs = dict(net_arch=[128, 64])


    if model_name == "PPO":
        model = PPO(PPOMlpPolicy, env, verbose=2, policy_kwargs=policy_kwargs, **hyperparam_kwargs) #policy_kwargs=policy_kwargs_ppo, **hyperparam_kwargs) 
    elif model_name == "SAC" and process_to_run != "train":
        model = SAC(SACMlpPolicy, env, verbose=2, **hyperparam_kwargs) # policy_kwargs=policy_kwargs_sac, **hyperparam_kwargs)   
    elif model_name == 'SAC' and process_to_run == "train":
        model = SAC(SACMlpPolicy, env, verbose=2, **hyperparam_kwargs)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    start_time = datetime.datetime.now()  # Record the start time

    model.learn(total_timesteps=steps)# , callback=tensorboard_callback)
    model_dir_path = os.path.join(MODEL_DIR, model_subdir)
    model_path = os.path.join(model_dir_path, f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    os.makedirs(model_dir_path, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    end_time = datetime.datetime.now()  # Record the end time
    duration = end_time - start_time  # Calculate the duration
    print(f"Training duration: {duration}")
    
    print(env.unwrapped.metadata)
    
    n_pursuers = env_kwargs["n_pursuers"]
    
    # Log file path
    log_file_path = os.path.join("logs", "tracking", f"train_{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}_a{n_pursuers}_{model_name}.txt")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Log env_kwargs, hyperparam_kwargs, and policy_kwargs
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Training on {str(env.unwrapped.metadata['name'])} with {model_choice} for {steps} steps in {steps/1000} episodes.\n")
        log_file.write("env_kwargs:\n")
        log_file.write(str(env_kwargs))
        log_file.write("\n\n")

        log_file.write("hyperparam_kwargs:\n")
        log_file.write(str(hyperparam_kwargs))
        log_file.write("\n\n")
        if model_name == "SAC":
            log_file.write("policy_kwargs_sac:\n")
            #log_file.write(str(policy_kwargs_sac))
            log_file.write("\n\n")
        elif model_name == "PPO":
            log_file.write("policy_kwargs_ppo:\n")
            log_file.write(str(policy_kwargs))
            log_file.write("\n\n")
        else:
            log_file.write("No policy_kwargs found.\n")
            log_file.write("\n\n")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Model saved to {model_path}\n")
        log_file.write("Model has been saved.\n")
        log_file.write(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")
        log_file.write(f"Training duration: {duration}\n")
           
    env.close()
    
def eval(env_fn, model_name, model_subdir=TRAIN_DIR, num_games=100, render_mode=None):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} with {model_name} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(
            glob.glob(os.path.join(MODEL_DIR, model_subdir, f"{env.metadata['name']}*.zip")), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    if model_name == "PPO":
        model = PPO.load(latest_policy)
    elif model_name == "SAC":
        model = SAC.load(latest_policy)
    elif model_name == "Heuristic":
        n_sensors = env_kwargs.get('n_sensors')
        sensor_range = env_kwargs.get('sensor_range')
    else:
        print("Invalid model name.")
        exit(0)

    total_rewards = {agent: 0 for agent in env.possible_agents}
    episode_avg_rewards = []

    for i in range(num_games):
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        env.reset(seed=i)
        agent_count = len(env.possible_agents)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            episode_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                if model_name == "Heuristic":
                    
                    action = communication_heuristic_policy(obs, n_sensors, sensor_range, agent_count, (int(agent[-1])+1))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    if model_name == "SAC":
                        action = action.reshape(env.action_space(agent).shape)
            env.step(action)

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg = sum(episode_rewards.values()) / len(episode_rewards)
        episode_avg_rewards.append(episode_avg)
        # print("episode_rewards", episode_rewards)
        # print("episode_avg", episode_avg)
        if i % 100 == 0:
            print(f"Rewards for episode {i}: {episode_rewards}")
            
        

    env.close()
    
    
    filename = f'pickles/eval_data_for_anova/{model_name}reward_data_e{num_games}_{current_datetime}_{env_kwargs["n_pursuers"]}.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)    
    with open(filename, 'wb') as f:
        pickle.dump(episode_avg_rewards, f)

    print(f"Data saved in {filename}")

    overall_avg_reward = sum(total_rewards.values()) / (len(total_rewards) * num_games)
    total_avg_reward = sum(episode_avg_rewards) 
    print("Total Rewards: ", total_rewards, "over", num_games, "games")
    print(f"Total Avg reward: {total_avg_reward}")
    print(f"Overall Avg reward: {overall_avg_reward}")
    
    if model_name == "PPO" or model_name == "SAC":
        # Get the latest log file
        log_files = glob.glob("logs/tracking/*.txt")
        log_file = max(log_files, key=os.path.getctime)
        
        with open(log_file, 'a') as file:
            file.write("Total Rewards: ")
            file.write(str(total_rewards))
            file.write(" over ")
            file.write(str(num_games))
            file.write(" games\n")
            file.write(f"Total Avg reward: {total_avg_reward}\n")
            file.write(f"Overall Avg reward: {overall_avg_reward}\n")

    

    return overall_avg_reward


def eval_with_model_path_run(env_fn, model_path, model_name, num_pursuers, sensor_range=None, poison_speed=None, sensor_count=None, num_games=100, render_mode=None):
    # Dynamic environment configuration based on the model specifics
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
                    action = communication_heuristic_policy(obs, n_sensors, env_kwargs['sensor_range'], len(env.possible_agents), (int(agent[-1])+1))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    
                # Store action, reward, and agent ID as a single array for later analysis
                actionid = np.append(action, [reward, int(agent[-1])])  # Ensure this matches the expected input format for Analysis
                actions.append(actionid)
                
            env.step(action)

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg_rewards.append(sum(episode_rewards.values()) / len(episode_rewards))

    env.close()

    overall_avg_reward = sum(total_rewards.values()) / (len(total_rewards) * num_games)
    print("Total Rewards: ", total_rewards, "over", num_games, "games")
    print(f"Overall Avg reward: {overall_avg_reward}")

    # Return both the actions and the overall average reward
    return {'actions': actions, 'overall_avg_reward': overall_avg_reward}





model_configs = [
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

def run_and_analyze_all_configs(games=100):
    all_results = {}  

    for config in model_configs:
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
            
            analysis = Analysis(data, output_dir=output_dir)
            analysis.full_analysis()
            
            print(f"Analysis results for {config['model_name']} with {config['n_pursuers']} pursuers: {analysis.results['evaluation']}")

            all_results[config_key] = {
                                            'data': data,
                                            'analysis_results': analysis.results,
                                            'avg_reward': avg_reward
                                        }
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




def run_advanced_analysis(config):
    # Configuration details for directory and file naming
    sensor_range_str = str(config["sensor_range"]).replace('.', 'p')  # Replace dot with 'p' for file system compatibility
    poison_speed_str = str(config["poison_speed"]).replace('.', 'p')
    config_suffix = f"sr{sensor_range_str}_ps{poison_speed_str}"

    # Setup the output directory with clear naming for easy identification and comparison
    output_dir = f"advanced_analysis_results/{current_datetime}" # /{config['model_name']}_pursuers_{config['n_pursuers']}_{config_suffix}"
    #check if the directory exists
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
        analysis.plot_hierarchical_clusters(n_clusters=5, plot_name=f"hierarchical_clusters_{config_suffix}.png")
        analysis.perform_time_frequency_analysis(plot_name=f"time_freq_analysis_{config_suffix}.png")
        analysis.plot_signal_histogram(plot_name=f"signal_histogram_{config_suffix}.png")
        analysis.create_k_distance_plot(plot_name=f"k_distance_plot_{config_suffix}.png")


        print(f"Completed advanced analysis for {config['model_name']} with {config['n_pursuers']} agents.")
    else:
        print("Failed to obtain results for advanced analysis.")



def eval_with_config(env_fn, model_path, model_name, num_pursuers, sensor_range=None, poison_speed=None, sensor_count=None, num_games=100, render_mode=None):
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
                    action = communication_heuristic_policy(obs, n_sensors, env_kwargs['sensor_range'], len(env.possible_agents), (int(agent[-1])+1))
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
    
    # Define the directory and file name dynamically
    directory = f'report/{current_datetime}'
    filename = f'rewards_{current_datetime}_{num_pursuers}.txt'
    path = os.path.join(directory, filename)

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Open the file in append mode and write content to it
    with open(path, 'a') as file:
        file.write(f'{model_name} with {num_pursuers} pursuers\n')
        file.write(f'{env_kwargs}\n')
        file.write(f"Total Rewards: {total_rewards} over {num_games} games\n")
        file.write(f"Overall Avg reward: {overall_avg_reward}\n")
    
    
    


    
    
    


def train_eval(env_fn, model, model_subdir, steps=100_000, seed=None, **hyperparam_kwargs):
    train_waterworld(env_fn, model, TRAIN_DIR, steps=steps, seed=0, **hyperparam_kwargs)
    eval(env_fn, model, num_games=1000, render_mode=None)

def save_communication_plot(communication_data, session_type):
    os.makedirs('scatterplot', exist_ok=True)

    signals, agent_ids = zip(*communication_data)
    plt.scatter(agent_ids, signals)
    plt.xlabel('Agent ID')
    plt.ylabel('Communication Signal')
    plt.title(f'{session_type} Session Communication Signals')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{session_type}_communication_plot_{current_time}.png"
    plt.savefig(f'scatterplot/{filename}')
    
def fine_tune_model(env_fn, model_name, model_subdir, model_path, steps=100_000, seed=None, **hyperparam_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 16, num_cpus=3, base_class="stable_baselines3")

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")

    if model_name == "PPO":
        model = PPO.load(model_path, env=env, **hyperparam_kwargs)
    elif model_name == "SAC":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Update the model with additional hyperparameters if needed
    for key, value in hyperparam_kwargs.items():
        setattr(model, key, value)

    print(f"Starting fine-tuning on {str(env.unwrapped.metadata['name'])}.")
    start_time = datetime.datetime.now()
    model.learn(total_timesteps=steps)
    end_time = datetime.datetime.now()

    # Saving the fine-tuned model
    fine_tuned_model_dir = os.path.join(MODEL_DIR, "fine_tuned", model_subdir)
    os.makedirs(fine_tuned_model_dir, exist_ok=True)
    fine_tuned_model_path = os.path.join(fine_tuned_model_dir, f"{model_name}_fine_tuned_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    model.save(fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")

    print(f"Finished fine-tuning on {str(env.unwrapped.metadata['name'])}. Duration: {end_time - start_time}")
    
    print(env.unwrapped.metadata)
    
    n_pursuers = env_kwargs["n_pursuers"]
    
    # Log file path
    log_file_path = os.path.join("logs", "tracking", f"train_{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}_finetune__a{n_pursuers}.txt")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Log env_kwargs, hyperparam_kwargs, and policy_kwargs
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Fine-tuning on {str(env.unwrapped.metadata['name'])} with {model_choice} for {steps} steps in {steps/1000} episodes.\n")
        log_file.write("env_kwargs:\n")
        log_file.write(str(env_kwargs))
        log_file.write("\n\n")

        log_file.write("hyperparam_kwargs:\n")
        log_file.write(str(hyperparam_kwargs))
        log_file.write("\n\n")
        

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Model saved to {fine_tuned_model_path}\n")
        log_file.write("Model has been saved.\n")
        log_file.write(f"Finished fine-tuning on {str(env.unwrapped.metadata['name'])}.\n")
        log_file.write(f"Fin-tuning duration: {end_time - start_time}\n")
        
    
    env.close()
    
    eval_with_model_path(env_fn, fine_tuned_model_path, model_name, num_games=1000, render_mode=None, analysis= False)
    #eval_with_model_path(env_fn, fine_tuned_model_path, model_name, num_games=1, render_mode="human", analysis= False)


def eval_with_model_path(env_fn, model_path, model_name, num_games=100, render_mode=None, analysis= False):
    actions=[]
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    if not os.path.exists(model_path):
        print("Model not found.")
        exit(0)

    if model_name == "PPO":
        model = PPO.load(model_path)
    elif model_name == "SAC":
        model = SAC.load(model_path)
    elif model_name == "Heuristic":
        n_sensors = env_kwargs.get('n_sensors')
        sensor_range = env_kwargs.get('sensor_range')
    else:
        print("Invalid model name.")
        exit(0)

    total_rewards = {agent: 0 for agent in env.possible_agents}
    episode_avg_rewards = []

    for i in range(num_games):
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            #print reward and reward type
            # print(reward, type(reward))
            
            episode_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                if model_name == "Heuristic":
                    # action = simple_policy(obs, n_sensors, sensor_range)
                    action = communication_heuristic_policy(obs, n_sensors, sensor_range, len(env.possible_agents))
                    actions.append((action, agent))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    actionid = np.append(action, [reward, int(agent[-1])])  # Append int(agent[-1]) to the action array
                    #print(actionid)
                    actions.append(actionid)
                    
                    if model_name == "SAC":
                        action = action.reshape(env.action_space(agent).shape)
                        actions.append((action, agent))
            env.step(action)

            

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg = sum(episode_rewards.values()) / len(episode_rewards)
        
        if i % 100 == 0:
            print(f"Rewards for episode {i}: {episode_rewards}")
        
        episode_avg_rewards.append(episode_avg)
        #print(f"Rewards for episode {i}: {episode_rewards}")

    env.close()
    
    # actions[i][0] = horizontal movement, actions[i][1] = vertical movement, actions[i][2] = communication signal

    overall_avg_reward = sum(total_rewards.values()) / (len(total_rewards) * num_games)
    total_avg_reward = sum(episode_avg_rewards) 
    print("Total Rewards: ", total_rewards, "over", num_games, "games")
    print(f"Total Avg reward: {total_avg_reward}")
    print(f"Overall Avg reward: {overall_avg_reward}")

    
    if analysis == True:
        
        base_dir = 'analysis_data'
        os.makedirs(base_dir, exist_ok=True)

        
        analysis_output_dir = os.path.join(base_dir, current_datetime)
        os.makedirs(analysis_output_dir, exist_ok=True)

        # Data preparation
        actions_array = np.array(actions)

        # Instantiate Analysis class with the output directory where results (including plots) will be saved
        analysis = Analysis(actions_array, analysis_output_dir)

        # Analysis Execution
        ## PCA and Regression
        analysis.apply_dynamic_pca()
        analysis.apply_pca_to_dependent_vars()
        analysis.regression_on_principal_components()

        ## Clustering
        analysis.apply_dbscan(eps=0.2, min_samples=2)
        analysis.apply_hierarchical_clustering()
        analysis.behavior_clustering(plot_name='behavior_clustering.png')

        ## Mutual Information and Entropy
        analysis.calculate_mutual_info_results()
        analysis.summarize_and_calculate_entropy(2)  
             
        
        ## GEE
        gee_results = analysis.apply_gee()

        ## Correlation
        communication_reward_correlation = analysis.calculate_correlation_with_performance()

        # Plotting Results
        analysis.plot_mutual_info_heatmap(plot_name='mutual_info_heatmap.png')
        analysis.plot_movement_communication_scatter(plot_name='movement_communication_scatter_plot.png')
        analysis.plot_pca_results(plot_name='pca_plot.png')
        analysis.plot_dbscan_results(plot_name='dbscan_clustering_plot.png')
        analysis.perform_time_frequency_analysis(plot_name='psd_plot.png')
        analysis.plot_signal_histogram(plot_name='signal_histogram.png')
        analysis.create_k_distance_plot(plot_name='k_distance_plot.png')
        analysis.save_results()
        
        # Save Analysis Results Textually
        results_file_path = os.path.join(analysis_output_dir, 'analysis_results.txt')
        with open(results_file_path, 'w') as file:
            # PCA and Regression
            file.write("PCA and Regression Results:\n")
            file.write(analysis.model_pc1.summary().as_text() + "\n\n")
            file.write(analysis.model_pc2.summary().as_text() + "\n\n")
            
            # Clustering
            file.write("Clustering Results:\n")
            file.write("DBSCAN and Hierarchical Clustering applied. See respective plot images.\n\n")
            
            # Mutual Information and Entropy
            file.write("Signal Summary:\n")
            com_sum = analysis.communication_summary
            file.write(f"Communication Signal Summary: {com_sum}\n")
            file.write("\n")
            file.write("Mutual Information and Entropy Results:\n")
            for result in analysis.mutual_info_results:
                file.write(f"Between Agent {result[0]} and Agent {result[1]}: MI = {result[2]}\n")
            file.write(f"\nEntropy all signals: {analysis.entropy_value}")
            individual_entropies = analysis.calculate_individual_agent_entropy()
            # Write entropies for individual agents
            file.write("\nEntropy of individual signals:\n")
            for agent_id, entropy_value in individual_entropies.items():
                file.write(f"Agent {agent_id} Entropy: {entropy_value}\n")
            
            file.write("\n")
            
            # GEE
            file.write("\nGEE Results:\n")
            for var, result in gee_results.items():
                file.write(f"{var}:\n{result.summary().as_text()}\n\n")
            
            # Correlation
            file.write("Correlation Analysis:\n")
            file.write(f"Correlation between Communication Signal and Performance: {communication_reward_correlation}\n")

        # Print Mutual Information Results to Console
        print("Mutual Information Results:")
        for result in analysis.mutual_info_results:
            print(f"Between Agent {result[0]} and Agent {result[1]}: MI = {result[2]}")
        
        

    

    return overall_avg_reward



# Train a model
def run_train(model='PPO'):
    episodes, episode_lengths = 20000, 1000
    total_steps = episodes * episode_lengths
    
    
    ppo_hyperparams = {
        'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Adaptive learning rate decreasing over epochs to fine-tune learning as it progresses. The lower bound ensures learning doesn't halt.
        'n_steps': 500,  # Ca n be increased to gather more experiences before each update, beneficial for complex environments with many agents and interactions. #org: 4096
        'batch_size': 128,  # Increased size to handle the complexity and data volume from multiple agents. Adjust based on computational resources.
        'n_epochs': 10,  # The number of epochs to run the optimization over the data. This remains standard but could be adjusted for finer tuning.
        'gamma': 0.998,  # Slightly higher to put more emphasis on future rewards, which is crucial in environments where long-term strategies are important.
        'gae_lambda': 0.92,  # Slightly lower to increase bias for more stable but potentially less accurate advantage estimates. Adjust based on variance in reward signals.
        'clip_range': lambda epoch: 0.1 + 0.15 / (1.0 + 0.1 * epoch), #lambda epoch: 0.1 + 0.15 * (0.98 ** epoch),  # Dynamic clipping range to gradually focus more on exploitation over exploration.
        'clip_range_vf': None,  # If None, clip_range_vf is set to clip_range. This could be set to a fixed value or a schedule similar to clip_range for value function clipping.
        'ent_coef': 0.005,  # Reduced to slightly decrease the emphasis on exploration as the agents' policies mature, considering the communication aspect.
        'vf_coef': 0.5,  # Remains unchanged; a balanced emphasis on the value function's importance is crucial for stable learning.
        'max_grad_norm': 0.5,  # Unchanged, as this generally provides good stability across a range of environments.
        'use_sde': True,  # Enables Stochastic Differential Equations for continuous action spaces, offering potentially smoother policy updates.
        'sde_sample_freq': 64,  # Determines how often to sample the noise for SDE, balancing exploration and computational efficiency.
        'normalize_advantage': True,  # Ensuring the advantages are normalized can improve learning stability and efficiency.
        
    }
       
    sac_hyperparams = {
        'learning_rate': 0.0003,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'action_noise': None,
        'replay_buffer_class': None,
        'replay_buffer_kwargs': None,
        'optimize_memory_usage': False,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'use_sde': False,
        'sde_sample_freq': -1,
        'use_sde_at_warmup': False,
        'stats_window_size': 100,
        'tensorboard_log': None,
        'policy_kwargs': None,
        #'verbose': 0,
        #'seed': None,
        'device': 'auto',
        '_init_setup_model': True
    }
    
    # sac_hyperparams = {
    #     'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Learning rate for Adam optimizer, affecting all networks (Q-values, Actor, and Value function). 
    #     'batch_size': 64,  # Size of minibatch for each gradient update, influencing the stability and speed of learning.
    #     'gamma': 0.998,  # Discount factor, impacting the present value of future rewards, closer to 1 makes future rewards more valuable.
    #     'tau': 0.005,  # Soft update coefficient for target networks, balancing between stability and responsiveness.
    #     'ent_coef': 'auto',  # Entropy regularization coefficient, 'auto' allows automatic adjustment for exploration-exploitation balance.
    #     'target_entropy': 'auto',  # Target entropy for automatic entropy coefficient adjustment, guiding exploration.
    #     'learning_starts': 500,  # Number of steps collected before learning starts, allowing for initial exploration.
    #     'buffer_size': 500,  # Size of the replay buffer, larger sizes allow for a more diverse set of experiences.
        
    #     'gradient_steps': -1,  # Matches the number of environment steps, ensuring comprehensive learning at each update.
    #     'optimize_memory_usage': False,  # While memory efficient, it can add complexity; consider based on available resources.
    #     'replay_buffer_class': None,  # Default replay buffer is generally sufficient unless specific modifications are needed.
    #     'replay_buffer_kwargs': None,  # Additional arguments for the replay buffer, if using a custom class.
    #     # 'use_sde': True,  # Enables generalized State Dependent Exploration (gSDE) for enhanced exploration.
    #     # 'sde_sample_freq': -1,  # Frequency of sampling new noise matrix for gSDE, -1 means only at the start of the rollout.
        
        
    #     'device': 'cuda',  # Utilizes GPU if available for faster computation.
        
    # }


    
    hyperparam_kwargs = ppo_hyperparams if model == 'PPO' else sac_hyperparams

    train_waterworld(env_fn, model, TRAIN_DIR, steps=total_steps, seed=0, **hyperparam_kwargs)
    eval(env_fn, model, num_games=1000, render_mode=None)
    #eval(env_fn, model, num_games=1, render_mode="human")
    
def run_eval(model='PPO'):
    eval(env_fn, model, num_games=1000, render_mode=None)

def run_eval_path(model='PPO',  path=r"models\train\waterworld_v4_20240314-050633.zip"): # models\train\waterworld_v4_20240301-081206.zip
    eval_with_model_path(env_fn, path, model, num_games=1, render_mode="human", analysis=False)
    #eval_with_model_path(env_fn, path, model, num_games=1, render_mode="human", analysis= False)

# Add a function to execute fine-tuning
def run_fine_tune(model='PPO', model_path=r"models\train\waterworld_v4_20240314-050633.zip"):
    episodes, episode_lengths = 20000, 1000
    total_steps = episodes * episode_lengths
    
    # ppo_hyperparams = {
    #     'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Adaptive learning rate decreasing over epochs to fine-tune learning as it progresses. The lower bound ensures learning doesn't halt.
    #     'n_steps': 600,  # Ca n be increased to gather more experiences before each update, beneficial for complex environments with many agents and interactions. #org: 4096
    #     'batch_size': 128,  # Increased size to handle the complexity and data volume from multiple agents. Adjust based on computational resources.
    #     'n_epochs': 10,  # The number of epochs to run the optimization over the data. This remains standard but could be adjusted for finer tuning.
    #     'gamma': 0.9999,  # Slightly higher to put more emphasis on future rewards, which is crucial in environments where long-term strategies are important.
    #     'gae_lambda': 0.92,  # Slightly lower to increase bias for more stable but potentially less accurate advantage estimates. Adjust based on variance in reward signals.
    #     'clip_range': lambda epoch: 0.1 + 0.15 / (1.0 + 0.1 * epoch), #lambda epoch: 0.1 + 0.15 * (0.98 ** epoch),  # Dynamic clipping range to gradually focus more on exploitation over exploration.
    #     'clip_range_vf': None,  # If None, clip_range_vf is set to clip_range. This could be set to a fixed value or a schedule similar to clip_range for value function clipping.
    #     'ent_coef': 0.005,  # Reduced to slightly decrease the emphasis on exploration as the agents' policies mature, considering the communication aspect.
    #     'vf_coef': 0.5,  # Remains unchanged; a balanced emphasis on the value function's importance is crucial for stable learning.
    #     'max_grad_norm': 0.5,  # Unchanged, as this generally provides good stability across a range of environments.
    #     'use_sde': True,  # Enables Stochastic Differential Equations for continuous action spaces, offering potentially smoother policy updates.
    #     'sde_sample_freq': 64,  # Determines how often to sample the noise for SDE, balancing exploration and computational efficiency.
    #     'normalize_advantage': True,  # Ensuring the advantages are normalized can improve learning stability and efficiency.
        
    # }
    
    ppo_hyperparams = {
        'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Adaptive learning rate decreasing over epochs to fine-tune learning as it progresses. The lower bound ensures learning doesn't halt.
        'n_steps': 500,  # Ca n be increased to gather more experiences before each update, beneficial for complex environments with many agents and interactions. #org: 4096
        'batch_size': 128,  # Increased size to handle the complexity and data volume from multiple agents. Adjust based on computational resources.
        'n_epochs': 10,  # The number of epochs to run the optimization over the data. This remains standard but could be adjusted for finer tuning.
        'gamma': 0.99,  # Slightly higher to put more emphasis on future rewards, which is crucial in environments where long-term strategies are important.
        'gae_lambda': 0.92,  # Slightly lower to increase bias for more stable but potentially less accurate advantage estimates. Adjust based on variance in reward signals.
        'clip_range': lambda epoch: 0.1 + 0.15 / (1.0 + 0.1 * epoch), #lambda epoch: 0.1 + 0.15 * (0.98 ** epoch),  # Dynamic clipping range to gradually focus more on exploitation over exploration.
        'clip_range_vf': None,  # If None, clip_range_vf is set to clip_range. This could be set to a fixed value or a schedule similar to clip_range for value function clipping.
        'ent_coef': 0.005,  # Reduced to slightly decrease the emphasis on exploration as the agents' policies mature, considering the communication aspect.
        'vf_coef': 0.5,  # Remains unchanged; a balanced emphasis on the value function's importance is crucial for stable learning.
        'max_grad_norm': 0.5,  # Unchanged, as this generally provides good stability across a range of environments.
        'use_sde': True,  # Enables Stochastic Differential Equations for continuous action spaces, offering potentially smoother policy updates.
        'sde_sample_freq': 64,  # Determines how often to sample the noise for SDE, balancing exploration and computational efficiency.
        'normalize_advantage': True,  # Ensuring the advantages are normalized can improve learning stability and efficiency.
        
    }
    

    sac_hyperparams = {
        'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),
        'batch_size': 256,
        'gamma': 0.999,
        'tau': 0.005,
        'ent_coef': 'auto',
        'target_entropy': 'auto',
        'use_sde': True,
        'sde_sample_freq': -1,
        'learning_starts': 500,
        'buffer_size': 1000,
        'gradient_steps': -1,
        'optimize_memory_usage': False,
        'replay_buffer_class': None,
        'replay_buffer_kwargs': None,
        'device': 'auto',
    }

    
    hyperparam_kwargs = ppo_hyperparams if model == 'PPO' else sac_hyperparams
    
    fine_tune_model(env_fn, model, "fine_tuned", model_path, steps=total_steps, seed=0, **hyperparam_kwargs)


if __name__ == "__main__":
    env_fn = waterworld_v4  
    process_to_run = 'eval_path'  # Options: 'train', 'optimize', 'eval', 'eval_path', 'analysis' or 'fine_tune'
    model_choice = 'Heuristic'  # Options: 'Heuristic', 'PPO', 'SAC'

    if model_choice == "Heuristic":
        process_to_run = 'eval'

    if process_to_run == 'train':
        run_train(model=model_choice)
    elif process_to_run == 'optimize':
        optimizer = GeneticHyperparamOptimizer(model_name=model_choice)
        best_hyperparams = optimizer.run(train_waterworld, eval, env_fn, population_size=20, generations=5)
        print("Best Hyperparameters:", best_hyperparams)
    elif process_to_run == 'eval':
        run_eval(model=model_choice)
    elif process_to_run == 'eval_path':
        run_eval_path(model=model_choice)
    elif process_to_run == 'fine_tune':
        run_fine_tune(model=model_choice, model_path=r"models\train\waterworld_v4_20240405-125529.zip")
    elif process_to_run == 'analysis':
        all_results = run_and_analyze_all_configs(games=evals)
        comparative_data = compare_across_configurations(all_results)
        comparative_data.to_csv(f'results/{current_datetime}/comparative_analysis.csv', index=False)
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
        
    elif process_to_run == 'advanced_analysis':
        advanced_configs = [
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
        
        for config in advanced_configs:
            run_advanced_analysis(config)
            
    elif process_to_run == 'multi_eval':
        config= [{"model_name": "Heuristic", "model_path": None, "n_pursuers": 4},
                {"model_name": "Heuristic", "model_path": None, "n_pursuers": 6},
                {"model_name": "Heuristic", "model_path": None, "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
                {"model_name": "Heuristic", "model_path": None, "n_pursuers": 2}]
            
        for config in config:
            eval_with_config(env_fn, config.get('model_path'), config.get('model_name'), config.get('n_pursuers'), config.get('sensor_range'), config.get('poison_speed'), config.get('sensor_count'), num_games=1000, render_mode=None)
    else:
        print("No Valid")
        
