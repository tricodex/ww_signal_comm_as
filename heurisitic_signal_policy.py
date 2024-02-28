import numpy as np

def enhanced_policy(observation, n_sensors, sensor_range, n_agents):
    # Extract sensor readings for food, poison
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]
    # Assuming communication signals are at the end of the observation
    communication_signals = observation[-n_agents:]

    # Determine action based on environmental stimuli
    action = np.array([0.0, 0.0, 0.0])  # [x, y, communication_signal]
    detection_threshold = 1 * sensor_range

    if np.any(food_dist < detection_threshold):
        closest_food_index = np.argmin(food_dist)
        action[:2] = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
        action[2] = 1.0  # Emit positive signal for food
    elif np.any(poison_dist < detection_threshold):
        closest_poison_index = np.argmin(poison_dist)
        action[:2] = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
        action[2] = -1.0  # Emit negative signal for poison
    else:
        # Consider signals from other agents
        if np.any(communication_signals > 0):  # Positive signal received
            # Move in the direction of the positive signal or continue exploring
            action[:2] = np.random.uniform(-1, 1, 2)  # Simplified decision
            action[2] = 0  # Neutral signal
        elif np.any(communication_signals < 0):  # Negative signal received
            # Avoid the direction of the negative signal
            action[:2] = np.random.uniform(-1, 1, 2)  # Simplified decision
            action[2] = 0  # Neutral signal
        else:
            # Random exploration
            random_direction = np.random.rand() * 2 * np.pi
            action[:2] = np.array([np.cos(random_direction), np.sin(random_direction)])
            action[2] = 0  # Neutral signal

    return np.array(action, dtype=np.float32)
