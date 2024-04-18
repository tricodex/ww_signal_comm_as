# heuristic_signal_policy.py

import numpy as np

def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents):
    # Extract all sensor readings from the observation vector
    sensor_data = {
        'obstacle_dist': observation[0:n_sensors],
        'barrier_dist': observation[n_sensors:2*n_sensors],
        'food_dist': observation[2*n_sensors:3*n_sensors],
        'food_speed': observation[3*n_sensors:4*n_sensors],
        'poison_dist': observation[4*n_sensors:5*n_sensors],
        'poison_speed': observation[5*n_sensors:6*n_sensors],
        'pursuer_dist': observation[6*n_sensors:7*n_sensors],
        'pursuer_speed': observation[7*n_sensors:8*n_sensors]
    }
    collision_food = observation[8 * n_sensors]
    collision_poison = observation[8 * n_sensors + 1]
    communication_signals = observation[-n_agents:]

    # Initialize action and communication signal
    action = np.zeros(2)  # Horizontal and vertical thrust
    communication_signal = 0  # CCS signal value

    # Define signal categories
    signal_thresholds = {
        'high_urgency': 0.75,
        'medium_urgency': 0.5,
        'low_urgency': 0.25,
        'neutral': 0,
        'resource_rich': -0.25,
        'resource_medium': -0.5,
        'resource_poor': -0.75
    }

    # Process communication signals
    signal_strength = np.mean(communication_signals)
    signal_type = 'neutral'
    for key, value in signal_thresholds.items():
        if signal_strength >= value:
            signal_type = key
            break

    # Strategy based on signal type and sensor data
    if 'urgency' in signal_type:
        # Handle urgency signals
        if 'high' in signal_type and np.any(sensor_data['poison_dist'] < sensor_range):
            closest_poison_index = np.argmin(sensor_data['poison_dist'])
            action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
            communication_signal = -1
        elif 'medium' in signal_type and np.any(sensor_data['pursuer_dist'] < sensor_range):
            # Medium urgency might relate to evading pursuers
            closest_pursuer_index = np.argmin(sensor_data['pursuer_dist'])
            action = -np.array([np.cos(closest_pursuer_index * 2 * np.pi / n_sensors),
                                np.sin(closest_pursuer_index * 2 * np.pi / n_sensors)])
            communication_signal = -0.5
    elif 'resource' in signal_type:
        # Handle resource signals
        if 'rich' in signal_type and np.any(sensor_data['food_dist'] < sensor_range):
            closest_food_index = np.argmin(sensor_data['food_dist'])
            action = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
            communication_signal = 1
        elif 'medium' in signal_type:
            # Explore towards average food location
            valid_food_indices = np.where(sensor_data['food_dist'] < sensor_range)[0]
            if valid_food_indices.size > 0:
                average_index = np.mean(valid_food_indices)
                action = np.array([np.cos(average_index * 2 * np.pi / n_sensors),
                                   np.sin(average_index * 2 * np.pi / n_sensors)])
                communication_signal = 0.5
    else:
        # No significant external signals; default to exploratory behavior
        random_direction = np.random.rand() * 2 * np.pi
        action = np.array([np.cos(random_direction), np.sin(random_direction)])
        communication_signal = 0  # Neutral signal

    return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)
