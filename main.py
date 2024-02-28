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

MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'
architecture = 256
 

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 128, use_extra_layers: bool = False, activation_function: str = 'relu', use_normalization: bool = False):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        input_dim = np.prod(observation_space.shape)
        layers = [nn.Linear(input_dim, features_dim)]
        
        if use_normalization:
            layers.append(nn.BatchNorm1d(features_dim))
        
        if activation_function == 'relu':
            layers.append(nn.ReLU())
        elif activation_function == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        
        if use_extra_layers:
            if use_normalization:
                layers.append(nn.BatchNorm1d(features_dim))
            layers.append(nn.Dropout(p=0.5))
            layers.append(nn.Linear(features_dim, features_dim))
            if activation_function == 'relu':
                layers.append(nn.ReLU())
            elif activation_function == 'leaky_relu':
                layers.append(nn.LeakyReLU())
        
        self._extractor = nn.Sequential(*layers)

    def forward(self, observations):
        return self._extractor(observations)
    

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging to Tensorboard.
    """
    def __init__(self, verbose=0, log_dir="tensorboard_logs"):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_training_start(self):
        log_dir = os.path.join(self.log_dir, f"{self.model.env.unwrapped.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
        self.logger = Monitor(self.model.env, log_dir) 

    def _on_step(self) -> bool:
        # Feature Evolution Tracking
        feature_values = self.model.policy.extract_features(self.model.rollout_buffer.observations[-1])
        self.logger.record('features/mean', feature_values.mean())
        self.logger.record('features/std', feature_values.std())
        self.logger.record('features/min', feature_values.min())  
        self.logger.record('features/max', feature_values.max())

        # Gradient Analysis
        for name, param in self.model.policy.named_parameters():
            if 'features_extractor' in name:
                self.logger.record(f'grads/{name}_norm', param.grad.norm().item())
            if 'policy' in name:  # Log gradients of the policy network as well
                self.logger.record(f'grads/{name}_norm', param.grad.norm().item())

        # Optional: Distribution Visualization
        plt.figure(figsize=(8, 4)) 
        plt.hist(feature_values.flatten(), bins=20)  
        plt.xlabel('Feature Values')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Values')
        self.logger.record('features/histogram', plt)

        # Optional: Correlation Analysis (if you have multiple features)
        if feature_values.shape[1] > 1:  # Check if you have multiple features
            corr_matrix = feature_values.corr()
            self.logger.record('features/corr_matrix', corr_matrix)

        return True


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
    log_dir = os.path.join("tensorboard_logs", f"{env.unwrapped.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}") 
    tensorboard_callback = TensorboardCallback(log_dir=log_dir)

       
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 20, num_cpus=6, base_class="stable_baselines3") # [1]=8
    
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
        "net_arch": {
            "pi": [architecture,],#  architecture,],# architecture],  # Actor network architecture
            "qf": [architecture, architecture]# architecture]   # Critic network architecture
        },
        "features_extractor_class": CustomFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "optimizer_class": th.optim.Adam,
        "optimizer_kwargs": {"weight_decay": 0.01},
        
        "share_features_extractor": True,
        "log_std_init": -2, 
        "ortho_init": False,  # Custom policy arguments, including initialization of log std and orthogonality of weights.
        
    }

    if model_name == "PPO":
        model = PPO(PPOMlpPolicy, env, verbose=3, **hyperparam_kwargs)#, policy_kwargs=policy_kwargs_ppo, **hyperparam_kwargs) 
    elif model_name == "SAC":
        
        

        model = SAC(SACMlpPolicy, env, verbose=3, policy_kwargs=policy_kwargs_sac, **hyperparam_kwargs)   
    elif model_name == 'SAC' and process_to_run == "train":
        model = SAC(SACMlpPolicy, env, verbose=3, buffer_size=10000 **hyperparam_kwargs)
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
    
    # Log file path
    log_file_path = os.path.join("logs", "tracking", f"trainings_{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.txt")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Log env_kwargs, hyperparam_kwargs, and policy_kwargs
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Training on {str(env.unwrapped.metadata['name'])} with {model_choice}.\n")
        log_file.write("env_kwargs:\n")
        log_file.write(str(env_kwargs))
        log_file.write("\n\n")

        log_file.write("hyperparam_kwargs:\n")
        log_file.write(str(hyperparam_kwargs))
        log_file.write("\n\n")

        log_file.write("policy_kwargs_sac:\n")
        log_file.write(str(policy_kwargs_sac))
        log_file.write("\n\n")

        log_file.write("policy_kwargs_ppo:\n")
        log_file.write(str(policy_kwargs_ppo))
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
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")

    if model_name == "PPO":
        model = PPO.load(model_path, env=env)
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
    fine_tuned_model_path = os.path.join(fine_tuned_model_dir, f"fine_tuned_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    model.save(fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")

    print(f"Finished fine-tuning on {str(env.unwrapped.metadata['name'])}. Duration: {end_time - start_time}")
    env.close()


def eval_with_model_path(env_fn, model_path, model_name, num_games=100, render_mode=None):
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

    # Plotting total rewards
    os.makedirs('plots/eval', exist_ok=True)
    plt.figure()
    plt.bar(total_rewards.keys(), total_rewards.values())
    plt.xlabel('Agents')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards per Agent in Waterworld Simulation')
    plot_name = f'{model_name}_rewards_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(f'plots/eval/{plot_name}')
    
    # #print shape and type of actions
    print("Shape of actions: ", np.shape(actions))
    print("Type of actions: ", type(actions))
    
    
    # Convert the list to a numpy array
    actions_array = np.array(actions)

    # Extract components
    horizontal_movements = actions_array[:, 0]
    vertical_movements = actions_array[:, 1]
    communication_signals = actions_array[:, 2]
    agent_ids = actions_array[:, 3]

    # Color-coded Scatter Plot based on Agent ID
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(horizontal_movements, vertical_movements, c=agent_ids, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Agent ID')
    plt.title('Movement Scatter Plot Color-coded by Agent ID')
    plt.xlabel('Horizontal Movement')
    plt.ylabel('Vertical Movement')
    plt.grid(True)
    plot_name = f'movement_scatter_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(f'plots/hvi/{plot_name}')
    plt.show()

    # 3D Scatter Plot incorporating Communication Signal
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(horizontal_movements, vertical_movements, communication_signals, c=agent_ids, cmap='viridis', alpha=0.5)
    fig.colorbar(scatter, ax=ax, label='Agent ID')
    ax.set_title('3D Scatter Plot of Movements and Communication Signal, Color-coded by Agent ID')
    ax.set_xlabel('Horizontal Movement')
    ax.set_ylabel('Vertical Movement')
    ax.set_zlabel('Communication Signal')
    plot_name = f'movement_communication_scatter_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(f'plots/hvsi/{plot_name}')
    plt.show()

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
    episodes, episode_lengths = 10000, 1000
    total_steps = episodes * episode_lengths
    
    
    ppo_hyperparams = {
        'learning_rate': lambda epoch: max(2.5e-4 * (0.85 ** epoch), 1e-5),  # Adaptive learning rate decreasing over epochs to fine-tune learning as it progresses. The lower bound ensures learning doesn't halt.
        'n_steps': 500,  # Ca n be increased to gather more experiences before each update, beneficial for complex environments with many agents and interactions. #org: 4096
        'batch_size': 128,  # Increased size to handle the complexity and data volume from multiple agents. Adjust based on computational resources.
        'n_epochs': 10,  # The number of epochs to run the optimization over the data. This remains standard but could be adjusted for finer tuning.
        'gamma': 0.995,  # Slightly higher to put more emphasis on future rewards, which is crucial in environments where long-term strategies are important.
        'gae_lambda': 0.92,  # Slightly lower to increase bias for more stable but potentially less accurate advantage estimates. Adjust based on variance in reward signals.
        'clip_range': lambda epoch: 0.1 + 0.15 * (0.98 ** epoch),  # Dynamic clipping range to gradually focus more on exploitation over exploration.
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
        'batch_size': 256,  # Size of minibatch for each gradient update, influencing the stability and speed of learning.
        'gamma': 0.999,  # Discount factor, impacting the present value of future rewards, closer to 1 makes future rewards more valuable.
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

def run_eval_path(model='PPO', path=r"models\train\waterworld_v4_20240228-144420.zip"):
    eval_with_model_path(env_fn, path, model, num_games=1, render_mode=None)
    #eval_with_model_path(env_fn, path, model, num_games=1, render_mode="human")

# Add a function to execute fine-tuning
def run_fine_tune(model='PPO', model_path=r"models\fine_tuned\fine_tuned\fine_tuned_20240228-100118.zip"):
    fine_tune_model(env_fn, model, "fine_tuned", model_path, steps=(98304*5), seed=0)



if __name__ == "__main__":
    env_fn = waterworld_v4  
    process_to_run = 'eval_path'  # Options: 'train', 'optimize', 'eval', 'eval_path' or 'fine_tune'
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
        run_fine_tune(model=model_choice, model_path=r"models\train\waterworld_v4_20240228-084753.zip")
        
