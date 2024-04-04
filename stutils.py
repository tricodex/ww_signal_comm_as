
import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

architecture = [256, 256]

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
    
    ############################################################################################################
    
    
    # def plot_residuals_histogram(self, plot_name='residuals_histogram.png'):
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(self.residuals, bins=20, edgecolor='k')
    #     plt.xlabel('Residuals')
    #     plt.ylabel('Frequency')
    #     plt.title('Histogram of Residuals')
    #     plt.savefig(os.path.join(self.output_dir, plot_name))  
    #     plt.close()

    # def plot_residuals_qq_plot(self, plot_name='residuals_qq_plot.png'):
    #     fig = sm.qqplot(self.residuals, line='45')
    #     plt.title('Q-Q Plot of Residuals')
    #     plt.savefig(os.path.join(self.output_dir, plot_name))  
    #     plt.close()
    
    # def plot_dendrogram(self, plot_name='dendrogram_plot.png'):
    #     plt.figure(figsize=(10, 7))
    #     dendrogram(self.linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    #     plt.title('Hierarchical Clustering Dendrogram')
    #     plt.xlabel('Sample Index')
    #     plt.ylabel('Distance')
    #     plt.savefig(os.path.join(self.output_dir, plot_name))  
    #     plt.close()
        
        
    # analysis.plot_residuals_histogram(plot_name='residuals_histogram.png')
    # analysis.plot_residuals_qq_plot(plot_name='residuals_qq_plot.png')
    # analysis.plot_dendrogram(plot_name='dendrogram_plot.png')
        
        
    # def plot_movement_communication_scatter(self, plot_name='movement_communication_scatter_plot.png'):
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     scatter = ax.scatter(self.df['Horizontal'], self.df['Vertical'], self.df['Communication'], c=self.df['AgentID'], cmap='viridis', alpha=0.5)
    #     fig.colorbar(scatter, ax=ax, label='Agent ID')
    #     ax.set_title('3D Scatter Plot of Movements and Communication Signal, Color-coded by Agent ID')
    #     ax.set_xlabel('Horizontal Movement')
    #     ax.set_ylabel('Vertical Movement')
    #     ax.set_zlabel('Communication Signal')
    #     plt.savefig(f'plots/hvsi/{plot_name}')
    #     plt.show()


    # # Get the current datetime
            # current_datetime = datetime.now()

            # # Create the visuals folder if it doesn't exist
            # folder_path = "visuals"
            # os.makedirs(folder_path, exist_ok=True)

            # # Save the recording with the current datetime as the filename
            # filename = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
            # file_path = os.path.join(folder_path, filename)
            
            
    # def determine_action(model, model_name, observation, env, agent):
#     if model_name == "Heuristic":
#         action = communication_heuristic_policy(observation)
#     else:
#         action, _states = model.predict(observation, deterministic=True)
#         if model_name == "SAC":
#             action = action.reshape(env.action_space(agent).shape)
#     return action

# def update_rewards_and_actions(total_rewards, actions, reward, action, agent):
#     for agent, agent_reward in reward.items():
#         total_rewards[agent] += agent_reward
        
#         actions.append((action, agent_reward, agent)) 
# Before wrapping the environment with SuperSuit
    # base_env = env_fn.parallel_env(**env_kwargs)
    # base_env_metadata = base_env.metadata  # Store the metadata from the base environment
    # base_env.reset(seed=seed)

    # # Now proceed with the wrapping
    # env = ss.pettingzoo_env_to_vec_env_v1(base_env)
    # env = ss.concat_vec_envs_v1(env, 12, num_cpus=2, base_class="stable_baselines3")

    # # Later, when needing to access or print metadata, use `base_env_metadata` instead of `env.metadata`
    # print(base_env_metadata)
    
    
    def step(self, action, agent_id, is_last):
        # Extract movement components from the action
        movement_action = action[:2] * self.pursuer_max_accel
        # Note: The communication part of the action is now determined by sensor readings

        # Limit thrust to max acceleration if necessary
        thrust = np.linalg.norm(movement_action)
        if thrust > self.pursuer_max_accel:
            movement_action = movement_action * (self.pursuer_max_accel / thrust)

        # Process movement for the current agent
        p = self.pursuers[agent_id]
        adjusted_velocity = np.clip(
            p.body.velocity + movement_action * self.pixel_scale,
            -self.pursuer_speed,
            self.pursuer_speed,
        )
        p.reset_velocity(adjusted_velocity[0], adjusted_velocity[1])

        # Update observation for the current state of the environment for this agent
        current_observation = self.observe(agent_id)  # This needs to be defined based on your environment

        # Use the current observation to determine the communication signal
        # Assuming the observation is structured with sensor readings followed by two collision indicators
        sensor_readings = current_observation[:8 * self.n_sensors]  # Adjust based on your actual observation structure
        p.emit_signal(sensor_readings)  # emit_signal now uses the real-time sensor readings to determine the signal

        # Store the communication signal for aggregation and interpretation
        self.communication_data[agent_id] = p.communication_signal

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * thrust

        if is_last:
            # Aggregate communication signals to determine collective behavior
            avg_signal = sum(self.communication_data) / len(self.communication_data)
            # Reset for next step
            self.communication_data = [0.0] * len(self.pursuers)

            # Apply environment physics update
            self.space.step(1 / self.FPS)

            # Adjust actions based on aggregated communication signal
            for id, other_pursuer in enumerate(self.pursuers):
                adjusted_velocity = self.adjust_velocity_based_on_signal(other_pursuer, avg_signal)
                other_pursuer.reset_velocity(adjusted_velocity[0], adjusted_velocity[1])

            # Calculate rewards considering adjusted behaviors
            for id in range(self.n_pursuers):
                other_pursuer = self.pursuers[id]
                self.behavior_rewards[id] = (
                    self.food_reward * other_pursuer.shape.food_indicator +
                    self.encounter_reward * other_pursuer.shape.food_touched_indicator +
                    self.poison_reward * other_pursuer.shape.poison_indicator
                )

                # Reset indicators for the next step
                other_pursuer.shape.food_indicator = 0
                other_pursuer.shape.poison_indicator = 0

            # Combine rewards and apply local/global reward strategy
            total_rewards = np.array(self.behavior_rewards) + np.array([accel_penalty] * self.n_pursuers)
            local_rewards = total_rewards
            global_rewards = local_rewards.mean()

            # Distribute rewards according to the local/global ratio
            self.last_rewards = local_rewards * self.local_ratio + global_rewards * (1 - self.local_ratio)

            self.frames += 1

        return self.observe(agent_id)