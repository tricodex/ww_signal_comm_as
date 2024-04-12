# signal_policy.py

import numpy as np

def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents):
    """
    A policy function that utilizes both local sensor data and communication signals from other agents
    to decide the next action in a simulated environment. This updated version includes enhanced processing
    of communication signals to determine movement more effectively.

    Args:
        observation (np.array): Array containing sensor and communication signal data.
        n_sensors (int): Number of sensors per agent.
        sensor_range (float): Maximum detection range of the sensors.
        n_agents (int): Number of agents that can send communication signals.

    Returns:
        np.array: Concatenated array of the calculated movement vector and communication signal.
    """
    # Extract sensor readings for food, poison, and communication signals from other agents
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]
    communication_signals = observation[-n_agents:]

    detection_threshold = sensor_range

    # Initialize action and communication signal
    action = np.array([0.0, 0.0])
    communication_signal = 0  # Neutral signal

    # Statistical analysis of communication signals
    signal_strength = np.sum(communication_signals) / np.count_nonzero(communication_signals) if np.any(communication_signals != 0) else 0

    # Decision-making enhanced by statistical signal processing
    if signal_strength > 0:  # Positive aggregate signal suggests moving towards potential resources
        if np.any(food_dist < detection_threshold):
            closest_food_index = np.argmin(food_dist)
            action = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
            communication_signal = 1
        else:
            # No immediate food detected, move based on overall positive signal trend
            direction = np.random.normal(loc=0, scale=1)
            action = np.array([np.cos(direction), np.sin(direction)])
            communication_signal = 0.5

    elif signal_strength < 0:  # Negative aggregate signal suggests avoiding potential hazards
        if np.any(poison_dist < detection_threshold):
            closest_poison_index = np.argmin(poison_dist)
            action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
            communication_signal = -1
        else:
            # No immediate poison detected, move based on overall negative signal trend
            direction = np.random.normal(loc=np.pi, scale=1)
            action = np.array([np.cos(direction), np.sin(direction)])
            communication_signal = -0.5

    else:
        # No significant signals received, base actions solely on local sensor data
        if np.any(food_dist < detection_threshold):
            closest_food_index = np.argmin(food_dist)
            action = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
            communication_signal = 1
        elif np.any(poison_dist < detection_threshold):
            closest_poison_index = np.argmin(poison_dist)
            action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
            communication_signal = -1
        else:
            # Random exploration when no clear data is available
            random_direction = np.random.rand() * 2 * np.pi
            action = np.array([np.cos(random_direction), np.sin(random_direction)])
            communication_signal = 0

    return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)


