# Visual of the network architecture for PPO and SAC algorithms

import torch
import torch.nn as nn
state_dimension = None
action_dimension = None

# PPO Network Architecture
# Input: State from the environment
# Output: Action in a continuous space

class PPO_MlpPolicy:
    def __init__(self):
        self.fc1 = nn.Linear(in_features=state_dimension, out_features=64)  # First fully connected layer
        self.relu1 = nn.ReLU()  # Activation function for the first layer
        self.fc2 = nn.Linear(in_features=64, out_features=64)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # Activation function for the second layer
        self.output = nn.Linear(in_features=64, out_features=action_dimension)  # Output layer for action

        # Output layer for action distribution's standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dimension))

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mean = self.output(x)
        log_std = self.log_std.expand_as(mean)
        std_dev = torch.exp(log_std)
        return mean, std_dev

# SAC Network Architecture
# Input: State from the environment
# Output: Action in a continuous space, Q-values for the critic

class SAC_MlpPolicy:
    def __init__(self):
        # Actor network
        self.actor_fc1 = nn.Linear(in_features=state_dimension, out_features=256)
        self.actor_relu1 = nn.ReLU()
        self.actor_fc2 = nn.Linear(in_features=256, out_features=256)
        self.actor_relu2 = nn.ReLU()
        self.actor_output = nn.Linear(in_features=256, out_features=action_dimension)
        
        # Critic network (twin critics)
        self.critic1_fc1 = nn.Linear(in_features=state_dimension + action_dimension, out_features=256)
        self.critic1_relu1 = nn.ReLU()
        self.critic1_fc2 = nn.Linear(in_features=256, out_features=256)
        self.critic1_relu2 = nn.ReLU()
        self.critic1_output = nn.Linear(in_features=256, out_features=1)

        self.critic2_fc1 = nn.Linear(in_features=state_dimension + action_dimension, out_features=256)
        self.critic2_relu1 = nn.ReLU()
        self.critic2_fc2 = nn.Linear(in_features=256, out_features=256)
        self.critic2_relu2 = nn.ReLU()
        self.critic2_output = nn.Linear(in_features=256, out_features=1)

    def forward(self, x, action=None):
        # Actor forward pass
        x = self.actor_relu1(self.actor_fc1(x))
        x = self.actor_relu2(self.actor_fc2(x))
        mean = self.actor_output(x)
        log_std = self.log_std.expand_as(mean)
        std_dev = torch.exp(log_std)
        
        # Critic forward pass (requires state and action as input)
        xa = torch.cat([x, action], dim=1)
        q1 = self.critic1_relu1(self.critic1_fc1(xa))
        q1 = self.critic1_relu2(self.critic1_fc2(q1))
        q1 = self.critic1_output(q1)

        q2 = self.critic2_relu1(self.critic2_fc1(xa))
        q2 = self.critic2_relu2(self.critic2_fc2(q2))
        q2 = self.critic2_output(q2)

        return mean, std_dev, q1, q2
