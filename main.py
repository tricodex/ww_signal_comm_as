# main.py

from __future__ import annotations
import glob
import os
import time
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

#from pettingzoo.sisl import waterworld_v4
import waterworld_v4 
from ga import GeneticHyperparamOptimizer
from settings import env_kwargs
import datetime
from heuristic_policy import simple_policy
from heurisitic_signal_policy import enhanced_policy

from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import gym
from gym import spaces
from torch import nn
import numpy as np

from analysis import Analysis

MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'
architecture = 256

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64, use_extra_layers: bool = True, activation_function: str = 'relu', use_normalization: bool = True):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        input_dim = np.prod(observation_space.shape)
        layers = [nn.Linear(input_dim, features_dim)]
        
        if use_normalization:
            layers.append(nn.BatchNorm1d(features_dim))
        layers.append(nn.ReLU() if activation_function == 'relu' else nn.LeakyReLU())
        
        if use_extra_layers:
            layers.extend([
                nn.Linear(features_dim, features_dim),
                nn.ReLU() if activation_function == 'relu' else nn.LeakyReLU()
            ])
        
        self._extractor = nn.Sequential(*layers)

    def forward(self, observations):
        return self._extractor(observations)

 

# class CustomFeaturesExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 128, use_extra_layers: bool = False, activation_function: str = 'relu', use_normalization: bool = False):
#         super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
#         input_dim = np.prod(observation_space.shape)
#         layers = [nn.Linear(input_dim, features_dim)]
        
#         if use_normalization:
#             layers.append(nn.BatchNorm1d(features_dim))
        
#         if activation_function == 'relu':
#             layers.append(nn.ReLU())
#         elif activation_function == 'leaky_relu':
#             layers.append(nn.LeakyReLU())
        
#         if use_extra_layers:
#             if use_normalization:
#                 layers.append(nn.BatchNorm1d(features_dim))
#             layers.append(nn.Dropout(p=0.5))
#             layers.append(nn.Linear(features_dim, features_dim))
#             if activation_function == 'relu':
#                 layers.append(nn.ReLU())
#             elif activation_function == 'leaky_relu':
#                 layers.append(nn.LeakyReLU())
        
#         self._extractor = nn.Sequential(*layers)

#     def forward(self, observations):
#         return self._extractor(observations)
    

# class TensorboardCallback(BaseCallback):
#     """
#     Custom callback for logging to Tensorboard.
#     """
#     def __init__(self, verbose=0, log_dir="tensorboard_logs"):
#         super(TensorboardCallback, self).__init__(verbose)
#         self.log_dir = log_dir

#     def _on_training_start(self):
#         log_dir = os.path.join(self.log_dir, f"{self.model.env.unwrapped.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
#         self.logger = Monitor(self.model.env, log_dir) 

#     def _on_step(self) -> bool:
#         # Feature Evolution Tracking
#         feature_values = self.model.policy.extract_features(self.model.rollout_buffer.observations[-1])
#         self.logger.record('features/mean', feature_values.mean())
#         self.logger.record('features/std', feature_values.std())
#         self.logger.record('features/min', feature_values.min())  
#         self.logger.record('features/max', feature_values.max())

#         # Gradient Analysis
#         for name, param in self.model.policy.named_parameters():
#             if 'features_extractor' in name:
#                 self.logger.record(f'grads/{name}_norm', param.grad.norm().item())
#             if 'policy' in name:  # Log gradients of the policy network as well
#                 self.logger.record(f'grads/{name}_norm', param.grad.norm().item())

#         # Optional: Distribution Visualization
#         plt.figure(figsize=(8, 4)) 
#         plt.hist(feature_values.flatten(), bins=20)  
#         plt.xlabel('Feature Values')
#         plt.ylabel('Frequency')
#         plt.title('Distribution of Feature Values')
#         self.logger.record('features/histogram', plt)

#         # Optional: Correlation Analysis (if you have multiple features)
#         if feature_values.shape[1] > 1:  # Check if you have multiple features
#             corr_matrix = feature_values.corr()
#             self.logger.record('features/corr_matrix', corr_matrix)

#         return True


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
    env = ss.concat_vec_envs_v1(env, 32, num_cpus=6, base_class="stable_baselines3") # [1]=8
    
    policy_kwargs_sac = {
        "net_arch": {
            "pi": [architecture,],#  architecture,],# architecture],  # Actor network architecture
            "qf": [architecture,  architecture,]# architecture]   # Critic network architecture
        },
        "features_extractor_class": CustomFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "optimizer_class": th.optim.Adam,
        "optimizer_kwargs": {"weight_decay": 0.01},
        "n_critics": 2,
        "share_features_extractor": True,
        
        
    }
    
    policy_kwargs_ppo = {
        "net_arch": [128, 64], #dict(pi=[128, 128], vf=[128, 128]),
        "features_extractor_class": CustomFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64, "use_extra_layers": True, "activation_function": "relu", "use_normalization": True},
        "optimizer_class": th.optim.Adam,
        "optimizer_kwargs": {"weight_decay": 0.01},
        
        "share_features_extractor": True,
        "log_std_init": -2, 
        "ortho_init": False,  # Custom policy arguments, including initialization of log std and orthogonality of weights.
        
    }

    if model_name == "PPO":
        model = PPO(PPOMlpPolicy, env, verbose=2, **hyperparam_kwargs) #policy_kwargs=policy_kwargs_ppo, **hyperparam_kwargs) 
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
    log_file_path = os.path.join("logs", "tracking", f"trainings_{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}_a:{n_pursuers}.txt")
    
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
            log_file.write(str(policy_kwargs_sac))
            log_file.write("\n\n")
        elif model_name == "PPO":
            log_file.write("policy_kwargs_ppo:\n")
            log_file.write(str(policy_kwargs_ppo))
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
    env = ss.concat_vec_envs_v1(env, 32, num_cpus=6, base_class="stable_baselines3")

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
    log_file_path = os.path.join("logs", "tracking", f"trainings_{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}_finetune__a:{n_pursuers}.txt")
    
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
    
    eval_with_model_path(env_fn, fine_tuned_model_path, model_name, num_games=10, render_mode=None, analysis= False)
    eval_with_model_path(env_fn, fine_tuned_model_path, model_name, num_games=1, render_mode="human", analysis= False)


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
            episode_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                if model_name == "Heuristic":
                    # action = simple_policy(obs, n_sensors, sensor_range)
                    action = enhanced_policy(obs, n_sensors, sensor_range, len(env.possible_agents))
                    actions.append((action, agent))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    actionid = np.append(action, int(agent[-1]))  # Append int(agent[-1]) to the action array
                    actions.append(actionid)
                    if model_name == "SAC":
                        action = action.reshape(env.action_space(agent).shape)
                        actions.append((action, agent))
            env.step(action)

            

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg = sum(episode_rewards.values()) / len(episode_rewards)
        episode_avg_rewards.append(episode_avg)
        print(f"Rewards for episode {i}: {episode_rewards}")

    env.close()
    
    # actions[i][0] = horizontal movement, actions[i][1] = vertical movement, actions[i][2] = communication signal

    overall_avg_reward = sum(total_rewards.values()) / (len(total_rewards) * num_games)
    total_avg_reward = sum(episode_avg_rewards) 
    print("Total Rewards: ", total_rewards, "over", num_games, "games")
    print(f"Total Avg reward: {total_avg_reward}")
    print(f"Overall Avg reward: {overall_avg_reward}")

    
    if analysis == True:

        # Assuming 'actions' is your dataset
        actions_array = np.array(actions)  # Convert your dataset to a numpy array

        # Instantiate the Analysis class with your actions array
        analysis = Analysis(actions_array)

        # Perform mutual information calculation between all pairs of agents
        analysis.calculate_mutual_info_results()

        # Print mutual information results to console
        analysis.print_mutual_info_results()

        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Optionally, save mutual information results to a file
        analysis.save_mutual_info_results(filepath=f'plots/analysis/mutual_information_results_{current_datetime}.txt')

        # Apply DBSCAN and Hierarchical clustering
        analysis.apply_dbscan(eps=0.5, min_samples=5)
        analysis.apply_hierarchical_clustering(method='ward')

        # Plot various analyses and visualizations
        analysis.plot_movement_scatter(plot_name=f'movement_scatter_plot_{current_datetime}.png')
        analysis.plot_movement_communication_scatter(plot_name=f'movement_communication_scatter_plot_{current_datetime}.png')
        analysis.plot_pca_results(plot_name=f'pca_plot_{current_datetime}.png')
        analysis.plot_clustering_results(plot_name=f'clustering_plot_{current_datetime}.png')
        analysis.plot_residuals_vs_predicted(plot_name=f'residuals_vs_predicted_plot_{current_datetime}.png')
        analysis.plot_residuals_histogram(plot_name=f'residuals_histogram_{current_datetime}.png')
        analysis.plot_residuals_qq_plot(plot_name=f'residuals_qq_plot_{current_datetime}.png')
        analysis.plot_dbscan_results(plot_name=f'dbscan_clustering_plot_{current_datetime}.png')
        analysis.plot_dendrogram(plot_name=f'dendrogram_plot_{current_datetime}.png')
        analysis.plot_mutual_info_heatmap(plot_name=f'mutual_info_heatmap_{current_datetime}.png')
        analysis.save_analysis_results(file_name=f'plots/analysis/analysis_results_{current_datetime}.txt')
        analysis.perform_time_frequency_analysis(plot_name = f'psd_plot_{current_datetime}.png')

    

    return overall_avg_reward

def eval(env_fn, model_name, model_subdir=TRAIN_DIR, num_games=100, render_mode=None):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

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
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            episode_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                if model_name == "Heuristic":
                    # action = simple_policy(obs, n_sensors, sensor_range)
                    action = enhanced_policy(obs, n_sensors, sensor_range, len(env.possible_agents))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    if model_name == "SAC":
                        action = action.reshape(env.action_space(agent).shape)
            env.step(action)

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg = sum(episode_rewards.values()) / len(episode_rewards)
        episode_avg_rewards.append(episode_avg)
        print(f"Rewards for episode {i}: {episode_rewards}")

    env.close()

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

    # Plotting total rewards
    os.makedirs('plots/eval', exist_ok=True)
    plt.figure()
    plt.bar(total_rewards.keys(), total_rewards.values())
    plt.xlabel('Agents')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards per Agent in Waterworld Simulation')
    plot_name = f'{model_name}_rewards_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(f'plots/eval/{plot_name}')

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
        'gamma': 0.9975,  # Slightly higher to put more emphasis on future rewards, which is crucial in environments where long-term strategies are important.
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
        'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Learning rate for Adam optimizer, affecting all networks (Q-values, Actor, and Value function). 
        'batch_size': 64,  # Size of minibatch for each gradient update, influencing the stability and speed of learning.
        'gamma': 0.998,  # Discount factor, impacting the present value of future rewards, closer to 1 makes future rewards more valuable.
        'tau': 0.005,  # Soft update coefficient for target networks, balancing between stability and responsiveness.
        'ent_coef': 'auto',  # Entropy regularization coefficient, 'auto' allows automatic adjustment for exploration-exploitation balance.
        'target_entropy': 'auto',  # Target entropy for automatic entropy coefficient adjustment, guiding exploration.
        'use_sde': True,  # Enables generalized State Dependent Exploration (gSDE) for enhanced exploration.
        'sde_sample_freq': -1,  # Frequency of sampling new noise matrix for gSDE, -1 means only at the start of the rollout.
        'learning_starts': 500,  # Number of steps collected before learning starts, allowing for initial exploration.
        'buffer_size': 500,  # Size of the replay buffer, larger sizes allow for a more diverse set of experiences.
        
        'gradient_steps': -1,  # Matches the number of environment steps, ensuring comprehensive learning at each update.
        'optimize_memory_usage': False,  # While memory efficient, it can add complexity; consider based on available resources.
        'replay_buffer_class': None,  # Default replay buffer is generally sufficient unless specific modifications are needed.
        'replay_buffer_kwargs': None,  # Additional arguments for the replay buffer, if using a custom class.
        
        
        'device': 'auto',  # Utilizes GPU if available for faster computation.
        
    }

    
    hyperparam_kwargs = ppo_hyperparams if model == 'PPO' else sac_hyperparams

    train_waterworld(env_fn, model, TRAIN_DIR, steps=total_steps, seed=0, **hyperparam_kwargs)
    eval(env_fn, model, num_games=10, render_mode=None)
    eval(env_fn, model, num_games=1, render_mode="human")
    
def run_eval(model='PPO'):
    eval(env_fn, model, num_games=1, render_mode="human")

def run_eval_path(model='PPO',  path=r"models\fine_tuned\fine_tuned\PPO_fine_tuned_20240308-201643.zip"): # models\train\waterworld_v4_20240301-081206.zip
    #eval_with_model_path(env_fn, path, model, num_games=10, render_mode=None, analysis= False)
    eval_with_model_path(env_fn, path, model, num_games=1, render_mode="human", analysis= False)
    


# Add a function to execute fine-tuning
def run_fine_tune(model='PPO', model_path=r"models\fine_tuned\fine_tuned\PPO_fine_tuned_20240308-201643.zip"):
    episodes, episode_lengths = 20000, 1500
    total_steps = episodes * episode_lengths
    
    ppo_hyperparams = {
        'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Adaptive learning rate decreasing over epochs to fine-tune learning as it progresses. The lower bound ensures learning doesn't halt.
        'n_steps': 600,  # Ca n be increased to gather more experiences before each update, beneficial for complex environments with many agents and interactions. #org: 4096
        'batch_size': 128,  # Increased size to handle the complexity and data volume from multiple agents. Adjust based on computational resources.
        'n_epochs': 10,  # The number of epochs to run the optimization over the data. This remains standard but could be adjusted for finer tuning.
        'gamma': 0.9999,  # Slightly higher to put more emphasis on future rewards, which is crucial in environments where long-term strategies are important.
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
    process_to_run = 'train'  # Options: 'train', 'optimize', 'eval', 'eval_path' or 'fine_tune'
    model_choice = 'PPO'  # Options: 'Heuristic', 'PPO', 'SAC'

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
        run_fine_tune(model=model_choice, model_path=r"models\fine_tuned\fine_tuned\PPO_fine_tuned_20240308-201643.zip")
        
