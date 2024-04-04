# run.py

from __future__ import annotations
import glob
import os
import supersuit as ss
from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy

#from pettingzoo.sisl import waterworld_v4
import waterworld_v4 
from ga import GeneticHyperparamOptimizer
from settings import env_kwargs
from utils import setup_logging, find_latest_model, make_json_serializable, print_evaluation_results, perform_detailed_analysis, log_experiment_results, generate_configurations, setup_experiment_logging, global_timestamp, get_configuration
from heurisitic_signal_policy import communication_heuristic_policy

import json
import logging
import argparse
import numpy as np
import gym
import ast


MODEL_DIR = os.path.join('models', 'train')
TRAIN_DIR = 'train' 
OPTIMIZE_DIR = 'optimize'
EXPERIMENT_BASE_DIR = 'experiments'

seed=42
num_envs=12 
num_cpus=2
env_fn = waterworld_v4

def select_model_and_policy(model_name, env, **kwargs):
    policy_kwargs = dict(net_arch=[128, 128])
    if model_name == 'PPO':
        model = PPO(PPOMlpPolicy, env, verbose=2, policy_kwargs=policy_kwargs, **kwargs)
    elif model_name == 'SAC':
        model = SAC(SACMlpPolicy, env, verbose=2, policy_kwargs=policy_kwargs, **kwargs)
    else:
        raise ValueError("Invalid model name. Please choose either 'PPO' or 'SAC'.")
    return model

def save_model_and_logs(model, model_name, operation, performance_metrics, model_params, training_params, env_name, model_dir):
    
    model_params = make_json_serializable(model_params)
    training_params = make_json_serializable(training_params)

    model_subdir = operation  

    model_dir_path = os.path.join(model_dir, model_subdir, f"{env_name}_{global_timestamp}")
    os.makedirs(model_dir_path, exist_ok=True)
    
    model_path = os.path.join(model_dir_path, f"{model_name}.zip")
    model.save(model_path)
    
    print(f"Model saved to {model_path}")
    logging.info(f"Model saved to {model_path}")

    
    metadata = {
        "model_name": model_name,
        "save_path": model_path,
        "performance_metrics": performance_metrics,
        "model_parameters": model_params,
        "training_parameters": training_params,
        "timestamp": global_timestamp
    }

    # Define the logs directory path where metadata will be saved
    logs_dir_path = os.path.join(model_dir_path, "logs")
    os.makedirs(logs_dir_path, exist_ok=True)  

    # Save metadata to a JSON file within the logs directory
    metadata_path = os.path.join(logs_dir_path, f"{model_name}_metadata_{global_timestamp}.json")
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print("Model and training metadata saved.")
    logging.info("Model and training metadata saved.")
    


def train_or_fine_tune_model(env_fn, model_name, steps, operation, env_kwargs=env_kwargs, seed=1, model_path=None, **hyperparam_kwargs):
    if model_name == "Heuristic":
        print("Heuristic policy selected. No training required.")
        return None 
    
    env = env_fn.parallel_env(**env_kwargs)  
    env.reset(seed=seed)
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    env_name = env.unwrapped.metadata.get('name')  
    n_pursuers = env_kwargs["n_pursuers"]  
    
    # Setup logging with dynamic information
    # setup_logging(env_name, n_pursuers, model_name, operation)

   
    if model_path:
        # Load the model for fine-tuning, apply hyperparameters
        model = load_model(model_path, model_name, env=env, fine_tune=True, **hyperparam_kwargs)
    else:
        # Training from scratch, no hyperparameters applied
        model = select_model_and_policy(model_name, env, **hyperparam_kwargs)
    
    model.learn(total_timesteps=steps)

    if not model_path:
        # Save the model after training if it's not fine-tuning
        model_dir_path = os.path.join(MODEL_DIR, operation, f"{env.unwrapped.metadata['name']}_{global_timestamp}")
        os.makedirs(model_dir_path, exist_ok=True)
        model_path = os.path.join(model_dir_path, f"{model_name}.zip")
        model.save(model_path)

    
    performance_metrics = {}
    model_params = get_configuration(model_name)  
    training_params = {
        "steps": steps,
        "seed": seed
    }
    
    save_model_and_logs(model, model_name, operation, performance_metrics, model_params, training_params, env_name, model_dir=MODEL_DIR)
    env.close()
    # Log training completion
    #logging.info("Training completed.")   
    return model_path
    
    
    

def evaluate_model(env_fn, model_name=None, model_path=None, num_games=100, render_mode=None, perform_analysis=False, env_kwargs=env_kwargs):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} with {model_name} (num_games={num_games}, render_mode={render_mode})")
    
    # Initialize model to None
    model = None

    if model_name != "Heuristic":
        if model_path is None:
            model_path = find_latest_model(MODEL_DIR, model_name)
            if not model_path:
                print("No available models found. Exiting.")
                return
        model = load_model(model_path, model_name, env)
        if model is None:
            print(f"Failed to load model: {model_path}")
            return
    else:
        print("Evaluating using heuristic policy. No model loading required.")

    env_kwargs.pop('model_path', None)
    print(f' model_path: {model_path}')
    #logging.info(f' model_path: {model_path}')
    print(f' model_name: {model_name}')
    #logging.info(f' model_name: {model_name}')
    print(f' num_games: {num_games}')
    #logging.info(f' num_games: {num_games}')
    print(f' render_mode: {render_mode}')
    #logging.info(f' render_mode: {render_mode}')
    print(f'Environment kwargs: {env_kwargs}')
    #logging.info(f'Environment kwargs: {env_kwargs}')

    # Ensure that the heuristic evaluation can proceed without a model
    if model_name == "Heuristic":
        total_rewards, actions = perform_evaluation(env, None, model_name, num_games)  # Pass None as the model for heuristic policies
    else:
        total_rewards, actions = perform_evaluation(env, model, model_name, num_games)

    if perform_analysis:
        perform_detailed_analysis(actions)

    print_evaluation_results(total_rewards, num_games)
    env.close()
    
    
    return total_rewards



def load_model(model_path, model_name, env=None, fine_tune=False, **hyperparam_kwargs):
    if model_name == "PPO":
        if fine_tune:
            model = PPO.load(model_path, env=env, **hyperparam_kwargs)
        else:
            model = PPO.load(model_path)
    elif model_name == "SAC":
        if fine_tune:
            model = SAC.load(model_path, env=env, **hyperparam_kwargs)
        else:
            model = SAC.load(model_path)
    elif model_name == "Heuristic":
        return
    else:
        print(f"Invalid model name:{model_name} does not exist. Exiting.")
        exit(0)
        
    return model

def perform_evaluation(env, model, model_name, num_games):
    total_rewards = {agent: 0 for agent in env.possible_agents}
    actions = []  
    episode_avg_rewards = []
    n_sensors = env_kwargs.get('n_sensors')
    sensor_range = env_kwargs.get('sensor_range')

    for i in range(num_games):
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        env.reset(seed=i) # use i for structured eval
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
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
                    #print(action)
                    actions.append(actionid)
                    if model_name == "SAC":
                        action = action.reshape(env.action_space(agent).shape)
                        actions.append((action, agent))
            env.step(action)

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg = sum(episode_rewards.values()) / len(episode_rewards)
        episode_avg_rewards.append(episode_avg)
        
    

    overall_avg_reward = sum(episode_avg_rewards) / num_games
    print("Evaluation completed. Overall average reward:", overall_avg_reward)

    return total_rewards, actions

# run_experiment 
def run_experiment(config, steps, num_games, env_kwargs):
    print(f"Running experiment with config: {config}")
    logging.info(f"Running experiment with config: {config}")
    model_name = config["algorithm"]
    env_kwargs_updated = {**env_kwargs, "n_pursuers": config.get("agents", env_kwargs.get("n_pursuers"))}
    
    # Adjusted to ensure model_path is determined here and passed to both train and evaluate functions
    model_path = None
    if config.get("model_path"):
        model_path = config["model_path"]
    else:
        # Potentially find the latest model here or leave it to the individual functions to handle
        model_path = find_latest_model(MODEL_DIR, model_name)
    # Ensuring model_path is included for training (if needed) and evaluation
    train_or_fine_tune_model(env_fn, model_name=model_name, steps=steps, env_kwargs=env_kwargs_updated, operation="train", seed=config.get("seed"), model_path=None)
    logging.info(f"Training completed for model: {model_name}")
    
    # Fixed the repetition of model_name keyword
    score = evaluate_model(env_fn, model_name=model_name, model_path=model_path, num_games=num_games, env_kwargs=env_kwargs_updated, render_mode=config.get("render_mode"), perform_analysis=config.get("perform_analysis"))
    logging.info(f"Evaluation completed for model: {model_name}")

    # Logging the results
    log_experiment_results(config, score, experiment_log_dir)
    logging.info("Experiment results logged")

    return {"config": config, "score": score}

   
    
def optimize_hyperparameters(model_name, env_fn):
    optimizer = GeneticHyperparamOptimizer(model_name=model_name)
    best_hyperparams = optimizer.run(
        train_function=train_or_fine_tune_model,
        eval_function=evaluate_model,
        env_fn=env_fn,
        population_size=20,
        generations=5,
        additional_args={'num_games': 10}  # Example of additional arguments that might be needed
    )
    print("Best Hyperparameters:", best_hyperparams)
    logging.info(f"Best Hyperparameters: {best_hyperparams}")
        
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="Model training, fine-tuning, evaluation, and experimentation script.")
    parser.add_argument("--mode", choices=["train", "fine_tune", "evaluate", "experiment"], help="Select the operation mode.")
    parser.add_argument("--model_name", help="Select the model type for non-experiment modes. Use --model_names for experiments.", default=None)
    parser.add_argument("--model_path", help="Path to the model file for fine-tuning or evaluation.", default=None)
    parser.add_argument("--num_games", type=int, help="Number of games for evaluation.", default=100)
    parser.add_argument("--render_mode", help="Render mode for evaluation.", default=None)
    parser.add_argument("--perform_analysis", action='store_true', help="Perform detailed analysis after evaluation.", default=False)
    parser.add_argument("--steps", type=int, help="Number of training steps for experiments.", default=100000)
    
    # Eval
    parser.add_argument("--env_kwargs", type=ast.literal_eval, help="Environment kwargs for evaluation.", default=env_kwargs)
    
    # Experiment 
    parser.add_argument("--model_names", nargs='+', help="List of model names for experiments.", default=["PPO"])
    parser.add_argument("--agent_counts", type=int, nargs='+', help="List of agent counts for experiments.", default=[2])    
    parser.add_argument("--include_no_comm", type=bool, nargs='?', const=False, default=False, help="Include configurations without communication. Defaults to True.")

    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    args = parse_arguments()
    if args.mode == "train":
        
        setup_logging('waterworld_v4', args.env_kwargs["n_pursuers"], args.model_name, mode="train")
        logging.info("Training started.")    
        logging.info(f"Environment kwargs: {args.env_kwargs}")    
        config = get_configuration(args.model_name)
        path = train_or_fine_tune_model(env_fn, model_name=args.model_name, steps=args.steps, env_kwargs=args.env_kwargs, operation="train", seed=config.get("seed"), model_path=None)
        logging.info("Training completed.")
        logging.info(f"Model saved to {path}")
        
    elif args.mode == "fine_tune" and args.model_path:
        
        setup_logging('waterworld_v4', args.env_kwargs["n_pursuers"], args.model_name, mode="fine_tune")
        logging.info("Fine-tuning started.")
        logging.info(f"Fine tuning on model path: {args.model_path}")
        logging.info(f"Model name: {args.model_name}")
        logging.info(f"Environment kwargs: {args.env_kwargs}")
        config = get_configuration(args.model_name)
        path = train_or_fine_tune_model(args.model_name, 
                                 model_path=args.model_path,  
                                 operation="fine_tune",
                                 **config)
        logging.info("Fine-tuning completed.")
        logging.info(f"Model saved to {path}")
        
    elif args.mode == "evaluate":
                
        setup_logging('waterworld_v4', args.env_kwargs["n_pursuers"], args.model_name, mode="evaluate")
        logging.info("Evaluation started.")
        logging.info(f"Evaluating on model path: {args.model_path}")
        logging.info(f"Model name: {args.model_name}")
        logging.info(f"Environment kwargs: {args.env_kwargs}")
        rewards = evaluate_model(env_fn, 
                       model_path=args.model_path, 
                       model_name=args.model_name, 
                       num_games=args.num_games, 
                       render_mode=args.render_mode, 
                       perform_analysis=args.perform_analysis,
                       env_kwargs=args.env_kwargs)
        logging.info("Evaluation completed.")
        logging.info(f"Total rewards: {rewards}")
        
    elif args.mode == "optimize":
        
        setup_logging('waterworld_v4', args.env_kwargs["n_pursuers"], args.model_name, mode="optimize")
        logging.info("Optimization started.")
        logging.info(f"Model name: {args.model_name}")
        logging.info(f"Environment kwargs: {args.env_kwargs}")
        
        
        optimizer = GeneticHyperparamOptimizer(model_name=args.model_name)
        best_hyperparams = optimizer.run(
            train_function=lambda **kwargs: train_or_fine_tune_model(args.model_name, 100000, env_fn, 1, None, **kwargs),
            eval_function=lambda model_path: evaluate_model(env_fn, model_path, args.model_name, num_games=100, render_mode=None, perform_analysis=False),
            env_fn=env_fn,
            population_size=20,
            generations=5
        )
        print("Best Hyperparameters found:", best_hyperparams)
        logging.info("Optimization completed.")
        logging.info(f"Best Hyperparameters found: {best_hyperparams}")
        
    elif args.mode == "experiment":
        experiment_log_dir = setup_experiment_logging()
        configurations = generate_configurations(args.agent_counts, args.model_names, args.include_no_comm)
        print(f"Running experiment with {len(configurations)} configurations.")
        logging.info(f"Running experiment with {len(configurations)} configurations.")
        print(f"Configurations: {configurations}")
        logging.info(f"Configurations: {configurations}")
        print(f"Steps: {args.steps}")
        logging.info(f"Steps: {args.steps}")
        print(f"Number of games: {args.num_games}")
        logging.info(f"Number of games: {args.num_games}")
        print(f"Environment kwargs: {env_kwargs}")
        logging.info(f"Environment kwargs: {env_kwargs}")
        results = []

        for config in configurations:
            config['steps'] = args.steps  # Apply the specified number of training steps
            env_kwargs['n_pursuers'] = config['agents']

            result = run_experiment(config, args.steps, args.num_games, env_kwargs)
            log_experiment_results(config, result['score'], experiment_log_dir)
            results.append(result)
            
        logging.info("Experiment completed.")
        logging.info("Results: {results}")
        
    else:
        print("Invalid mode selected. Exiting.")
        exit(0)
        
        
        
# python run.py --mode experiment --model_names PPO SAC Heuristic --agent_counts 2 --steps 200 --num_games 1

# python run.py --mode evaluate --model_name PPO --model_path models\train\waterworld_v4_20240314-050633.zip --num_games 1 --render_mode human --env_kwargs "{'n_pursuers': 6, 'n_evaders': 6, 'n_poisons': 8, 'n_coop': 2, 'n_sensors': 16, 'sensor_range': 0.2, 'radius': 0.015, 'obstacle_radius': 0.055, 'n_obstacles': 1, 'obstacle_coord': [(0.5, 0.5)], 'pursuer_max_accel': 0.01, 'evader_speed': 0.01, 'poison_speed': 0.075, 'poison_reward': -10, 'food_reward': 1070.0, 'encounter_reward': 0.015, 'thrust_penalty': -0.01, 'local_ratio': 0.0, 'speed_features': True, 'max_cycles': 1000}"

# python run.py --mode evaluate --model_name PPO --model_path models\train\train\waterworld_v4_20240401-233948\PPO.zip --num_games 1 --render_mode human  

# python run.py --mode train --model_name PPO        
            
