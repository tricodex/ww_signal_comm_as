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
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

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
    
    rewards = {agent: 0 for agent in env.possible_agents}

    


    if model_name == "PPO":
        model = PPO.load(latest_policy)
        for i in range(num_games):
            env.reset(seed=i)
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                for a in env.agents:
                    rewards[a] += env.rewards[a]

                    
                
                if termination or truncation:
                    break
                else:
                    act = model.predict(obs, deterministic=True)[0]
                env.step(act)
        env.close()

    elif model_name == "SAC":
        model = SAC.load(latest_policy)
        for i in range(num_games):
            env.reset(seed=i)
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                for a in env.agents:
                    rewards[a] += env.rewards[a]

                    
                
                if termination or truncation:
                    action = None
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    action = action.reshape(env.action_space(agent).shape) if model_name == "SAC" else action
                env.step(action)
        env.close()
        
    elif model_name == "Heuristic":  # Add a condition for heuristic policy
        n_sensors = env_kwargs.get('n_sensors')  
        sensor_range = env_kwargs.get('sensor_range')
        for i in range(num_games):
            env.reset(seed=i)
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action = simple_policy(obs, n_sensors, sensor_range)
                env.step(action)
                rewards[agent] += reward  # Update rewards after action step

                

        env.close()
    
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")


    

    # Plotting total rewards
    os.makedirs('plots/eval', exist_ok=True)
    if num_games == 10: # maybe change this
        plt.figure()
        plt.bar(rewards.keys(), rewards.values())
        plt.xlabel('Agents')
        plt.ylabel('Total Rewards')
        plt.title('Total Rewards per Agent in Waterworld Simulation')
        plot_name = f'{mdl}_{process_to_run}_rewards_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
        plt.savefig(f'plots/eval/{plot_name}')

    return avg_reward

# Train a model
def run_train():
    # still arbitrary episodes and episode lengths
    episodes, episode_lengths = 2, 98304
    total = episode_lengths*episodes
        
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
    
    # Train the waterworld environment with the specified model, settings, and hyperparameters
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=total, seed=0, **hyperparam_kwargs)
    
    # Evaluate the trained model against a random agent for 10 games without rendering
    eval(env_fn, mdl, num_games=10, render_mode=None)
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    eval(env_fn, mdl, num_games=1, render_mode="human")
    
def run_eval():
    # Evaluate the trained model against a random agent for 10 games without rendering
    eval(env_fn, mdl, num_games=10, render_mode="human")
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    # eval(env_fn, mdl, num_games=1, render_mode="human")

def quick_test():
    # Train the waterworld environment with the specified model and settings
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=1, seed=0)
    
    # Evaluate the trained model against a random agent for 10 games without rendering
    eval(env_fn, mdl, num_games=10, render_mode=None)
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    eval(env_fn, mdl, num_games=1, render_mode="human")

if __name__ == "__main__":
    env_fn = waterworld_v4  
    process_to_run = 'train'  # Choose "train", "optimize", "optimize_parallel" or "eval"
    mdl = "PPO"# Choose "Heuristic", "PPO" or "SAC"
    
    # security check
    if mdl == "Heuristic":
        process_to_run = 'eval'

    if process_to_run == 'train':
        run_train()
    elif process_to_run == 'optimize':
        optimizer = GeneticHyperparamOptimizer(model_name=mdl)
        best_hyperparams = optimizer.run(
            train_waterworld, 
            eval, 
            env_fn, 
            population_size=30,
            generations=20
        )
        print("Best Hyperparameters:", best_hyperparams)
    # elif process_to_run == 'optimize_parallel':
    #     optimizer = GeneticHyperparamOptimizer(model_name=mdl)
    #     best_hyperparams = optimizer.run_parallel(
    #         train_waterworld_parallel,  
    #         eval, 
    #         env_fn, 
    #         population_size=4,
    #         generations=4
    #     )
    #     print("Best Hyperparameters:", best_hyperparams)
    
    elif process_to_run == 'eval':
        run_eval()
    elif process_to_run == 'qt':
        quick_test()
        

    # INFO:root:Evaluating Individual: {'learning_rate': 0.0008637620730511329, 'batch_size': 115.73571005105222, 'gamma': 0.8, 'gae_lambda': 0.9087173146398605, 'n_steps': 1024, 'ent_coef': 0.00010643155693475564, 'vf_coef': 0.9533843157531028, 'max_grad_norm': 6.051664754689212, 'clip_range': 0.7697781042380724}, Avg Reward: 235.44880737923336
