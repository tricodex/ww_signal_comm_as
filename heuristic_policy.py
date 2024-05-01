# heuristic_policy.py

import numpy as np

def simple_policy(observation, n_sensors, sensor_range):
    

    # Extract sensor readings for food and poison
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]

    # Threshold to consider an object "detected"
    detection_threshold = 1 # no matter the env sensor range, the actual distance range is always scaled down to [0,1]

    # Initialize action
    action = np.array([0.0, 0.0])

    # Check for food and poison
    if np.any(food_dist < detection_threshold):
        # Move towards the closest food
        closest_food_index = np.argmin(food_dist)
        action[0] = np.cos(closest_food_index * 2 * np.pi / n_sensors)
        action[1] = np.sin(closest_food_index * 2 * np.pi / n_sensors)
    elif np.any(poison_dist < detection_threshold):
        # Move away from the closest poison
        closest_poison_index = np.argmin(poison_dist)
        action[0] = -np.cos(closest_poison_index * 2 * np.pi / n_sensors)
        action[1] = -np.sin(closest_poison_index * 2 * np.pi / n_sensors)
    else:
        # Random wander
        random_direction = np.random.rand() * 2 * np.pi
        action[0] = np.cos(random_direction)
        action[1] = np.sin(random_direction)

    # Assuming 'action' is the variable holding your calculated action
    action = np.array(action, dtype=np.float32)


    return action

