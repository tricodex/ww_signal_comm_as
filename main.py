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
#from stable_baselines3.common.monitor import Monitor
#from pettingzoo.sisl import waterworld_v4
import waterworld_v4 
from ga import GeneticHyperparamOptimizer
from settings import env_kwargs
import datetime
from heuristic_policy import simple_policy


MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'

def train_waterworld(env_fn, model_name, model_subdir, steps=100_000, seed=None, **hyperparam_kwargs):
    
    if 'n_steps' in hyperparam_kwargs:
        hyperparam_kwargs['n_steps'] = int(hyperparam_kwargs['n_steps'])
    if 'batch_size' in hyperparam_kwargs:
        hyperparam_kwargs['batch_size'] = int(hyperparam_kwargs['batch_size'])
    if 'buffer_size' in hyperparam_kwargs:
        hyperparam_kwargs['buffer_size'] = int(hyperparam_kwargs['buffer_size'])
    
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    
    
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3") # [1]=8

    # # Specify the file path for Monitor to save data
    # monitor_dir = "./monitors/"  
    # os.makedirs(monitor_dir, exist_ok=True)
    # env = Monitor(env, os.path.join(monitor_dir, "monitor.csv"))


    if model_name == "PPO":
        model = PPO(PPOMlpPolicy, env, verbose=3, **hyperparam_kwargs) # tensorboard_log="./ppo_tensorboard/",
    elif model_name == "SAC":
        #policy_kwargs = {"net_arch": [dict(pi=[400, 300], qf=[400, 300])]} # policy_kwargs=policy_kwargs
        model = SAC(SACMlpPolicy, env, verbose=3, **hyperparam_kwargs)   #, tensorboard_log="./sac_tensorboard/"
    elif model_name == 'SAC' and process_to_run == "train":
        model = SAC(SACMlpPolicy, env, verbose=3, buffer_size=10000 **hyperparam_kwargs)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    start_time = datetime.datetime.now()  # Record the start time

    model.learn(total_timesteps=steps)
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

def eval_with_model_path(env_fn, model_path, model_name, num_games=100, render_mode=None):
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
                    action = simple_policy(obs, n_sensors, sensor_range)
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    if model_name == "SAC":
                        action = action.reshape(env.action_space(agent).shape)
            env.step(action)

            if 'communication_data' in info:
                communication_data = info['communication_data']
                save_communication_plot(communication_data, session_type="Eval")

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
                    action = simple_policy(obs, n_sensors, sensor_range)
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
def run_train(ppos=True, sacs=False):
    # still arbitrary episodes and episode lengths
    episodes, episode_lengths = 3, 98304
    total = episode_lengths*episodes
    
    if ppos == True:    
        # Specify the hyperparameters
        hyperparam_kwargs = {
            'learning_rate': 0.0008637620730511329,
            'batch_size': 115.73571005105222,
            'gamma': 0.8,
            'gae_lambda': 0.9087173146398605,
            'n_steps': 1024,
            'ent_coef': 0.00010643155693475564,
            'vf_coef': 0.9533843157531028,
            'max_grad_norm': 6.051664754689212,
            'clip_range': 0.7697781042380724
        }
    elif sacs == True:
        hyperparam_kwargs = {
            'learning_rate': 0.001,
            'batch_size': 526.2398081354436,
            'gamma': 0.9805953886911123,
            'tau': 0.01922034521763273,
            'ent_coef': 0.01,
            'target_entropy': -0.03305947857437355,
            'use_sde': 0.9186635228704448,
            'sde_sample_freq': 1.1470795165037608,
            'learning_starts': 10000,
            'buffer_size': 18921.859374917258
        }   
    else:
        hyperparam_kwargs ={}
    
    # Train the waterworld environment with the specified model, settings, and hyperparameters
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=total, seed=0, **hyperparam_kwargs)
    
    # Evaluate the trained model against a random agent for 10 games without rendering
    eval(env_fn, mdl, num_games=10, render_mode=None)
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    eval(env_fn, mdl, num_games=1, render_mode="human")
    
def run_eval():
    # Evaluate the trained model against a random agent for 10 games without rendering
    # eval(env_fn, mdl, num_games=10, render_mode="human")
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    eval(env_fn, mdl, num_games=1, render_mode="human")
    
def run_eval_path():
    path = r"models\train\waterworld_v4_20240223-135726.zip"
    eval_with_model_path(env_fn, path, mdl, num_games=1, render_mode="human")

def quick_test():
    # Train the waterworld environment with the specified model and settings
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=1, seed=0)
    
    # Evaluate the trained model against a random agent for 10 games without rendering
    eval(env_fn, mdl, num_games=10, render_mode=None)
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    eval(env_fn, mdl, num_games=1, render_mode="human")

if __name__ == "__main__":
    env_fn = waterworld_v4  
    process_to_run = 'train'  # Choose "train", "optimize" or "eval"
    mdl = "PPO"# Choose "Heuristic", "PPO" or "SAC"
    
    # security check
    if mdl == "Heuristic":
        process_to_run = 'eval_path'

    if process_to_run == 'train':
        run_train()#ppos=False, sacs=True)
    elif process_to_run == 'optimize':
        optimizer = GeneticHyperparamOptimizer(model_name=mdl)
        best_hyperparams = optimizer.run(
            train_waterworld, 
            eval, 
            env_fn, 
            population_size=8,
            generations=5
        )
        print("Best Hyperparameters:", best_hyperparams)
    
    elif process_to_run == 'eval':
        run_eval()
    elif process_to_run == 'qt':
        quick_test()
    elif process_to_run == 'eval_path':
        run_eval_path()
        
