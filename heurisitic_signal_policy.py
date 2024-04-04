import numpy as np

def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents):
    # Extract sensor readings for food, poison, and communication signals from other agents
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]
    communication_signals = observation[-n_agents:]

    detection_threshold = sensor_range

    # Initialize action and communication signal
    action = np.array([0.0, 0.0])
    communication_signal = 0  # Neutral signal

    # Enhance decision-making by incorporating communication signals
    if np.any(communication_signals > 0):  # Positive signal received from others
        # Prioritize moving towards the direction with the strongest positive signal if food is also detected within range
        if np.any(food_dist < detection_threshold):
            closest_food_index = np.argmin(food_dist)
            action = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
            communication_signal = 1
        else:
            # Move in a direction based on positive communication if no immediate food is detected
            action = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            communication_signal = 0.5  # Indicate a move towards a potential food source based on communication

    elif np.any(communication_signals < 0):  # Negative signal received from others
        # Avoid areas where poison might be, based on communication, unless poison is directly detected and must be avoided
        if np.any(poison_dist < detection_threshold):
            closest_poison_index = np.argmin(poison_dist)
            action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
            communication_signal = -1
        else:
            # Random exploration with a slight avoidance bias if only negative signals received
            action = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            communication_signal = -0.5  # Indicate caution due to negative signals

    else:
        # Basic decision-making based on local environment without communication input
        if np.any(food_dist < detection_threshold):
            closest_food_index = np.argmin(food_dist)
            action = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
            communication_signal = 1  # Positive signal for food
        elif np.any(poison_dist < detection_threshold):
            closest_poison_index = np.argmin(poison_dist)
            action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
            communication_signal = -1  # Negative signal for poison
        else:
            # Random exploration
            random_direction = np.random.rand() * 2 * np.pi
            action = np.array([np.cos(random_direction), np.sin(random_direction)])
            communication_signal = 0  # Neutral signal, as the agent is exploring

    return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)
    
    
    
    
    # # Extract sensor readings for food, poison
    # food_dist = observation[2 * n_sensors:3 * n_sensors]
    # poison_dist = observation[4 * n_sensors:5 * n_sensors]
    # # Assuming communication signals are at the end of the observation
    # communication_signals = observation[-n_agents:]

    # # Determine action based on environmental stimuli
    # action = np.array([0.0, 0.0, 0.0])  # [x, y, communication_signal]
    # detection_threshold = 1 * sensor_range

    # if np.any(food_dist < detection_threshold):
    #     closest_food_index = np.argmin(food_dist)
    #     action[:2] = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
    #                            np.sin(closest_food_index * 2 * np.pi / n_sensors)])
    #     action[2] = 1.0  # Emit positive signal for food
    # elif np.any(poison_dist < detection_threshold):
    #     closest_poison_index = np.argmin(poison_dist)
    #     action[:2] = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
    #                             np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
    #     action[2] = -1.0  # Emit negative signal for poison
    # else:
    #     # Consider signals from other agents
    #     if np.any(communication_signals > 0):  # Positive signal received
    #         # Move in the direction of the positive signal or continue exploring
    #         action[:2] = np.random.uniform(-1, 1, 2)  # Simplified decision
    #         action[2] = 0  # Neutral signal
    #     elif np.any(communication_signals < 0):  # Negative signal received
    #         # Avoid the direction of the negative signal
    #         action[:2] = np.random.uniform(-1, 1, 2)  # Simplified decision
    #         action[2] = 0  # Neutral signal
    #     else:
    #         # Random exploration
    #         random_direction = np.random.rand() * 2 * np.pi
    #         action[:2] = np.array([np.cos(random_direction), np.sin(random_direction)])
    #         action[2] = 0  # Neutral signal

    # return np.array(action, dtype=np.float32)
