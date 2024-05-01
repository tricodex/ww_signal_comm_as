# heuristic_signal_policy.py

import numpy as np


import numpy as np

def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents):
    # Extract sensor data from observation vector
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

    # Initialize action and communication signal
    action = np.zeros(2)  # Horizontal and vertical thrust
    communication_signal = 0  # Default to neutral

    # Process dynamic threshold and signal type
    poison_distances = sensor_data['poison_dist'][sensor_data['poison_dist'] < sensor_range]
    pursuer_distances = sensor_data['pursuer_dist'][sensor_data['pursuer_dist'] < sensor_range]
    average_poison_dist = np.mean(poison_distances) if poison_distances.size > 0 else float('inf')
    average_pursuer_dist = np.mean(pursuer_distances) if pursuer_distances.size > 0 else float('inf')

    dynamic_thresholds = {
        'high_urgency': 0.2 if np.isinf(average_poison_dist) else max(0.2, 1 - average_poison_dist/sensor_range),
        'medium_urgency': 0.1 if np.isinf(average_pursuer_dist) else max(0.1, 1 - average_pursuer_dist/sensor_range)
    }

    signal_strength = np.mean(observation[-n_agents:])

    signal_type = 'neutral'
    for key, value in dynamic_thresholds.items():
        if signal_strength >= value:
            signal_type = key
            break

    # Define actions based on signal type
    if 'urgency' in signal_type:
        if 'high' in signal_type:
            #print('High S')
            closest_poison_index = np.argmin(sensor_data['poison_dist'])
            action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
                                np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
            communication_signal = -1
        elif 'medium' in signal_type:
            
            closest_pursuer_index = np.argmin(sensor_data['pursuer_dist'])
            action = -np.array([np.cos(closest_pursuer_index * 2 * np.pi / n_sensors),
                                np.sin(closest_pursuer_index * 2 * np.pi / n_sensors)])
            communication_signal = -0.5
    elif signal_type == 'neutral' and np.isinf(average_poison_dist) and np.isinf(average_pursuer_dist):
        # Fallback proactive or preventive behavior when no threats are detected
        if np.any(sensor_data['food_dist'] < sensor_range):
            closest_food_index = np.argmin(sensor_data['food_dist'])
            action = np.array([np.cos(closest_food_index * 2 * np.pi / n_sensors),
                               np.sin(closest_food_index * 2 * np.pi / n_sensors)])
            communication_signal = 0.5  # Signal to attract to food sources as a preventive strategy
            
    return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)


# def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents):
#     # Extract all sensor readings from the observation vector
#     sensor_data = {
#         'obstacle_dist': observation[0:n_sensors],
#         'barrier_dist': observation[n_sensors:2*n_sensors],
#         'food_dist': observation[2*n_sensors:3*n_sensors],
#         'food_speed': observation[3*n_sensors:4*n_sensors],
#         'poison_dist': observation[4*n_sensors:5*n_sensors],
#         'poison_speed': observation[5*n_sensors:6*n_sensors],
#         'pursuer_dist': observation[6*n_sensors:7*n_sensors],
#         'pursuer_speed': observation[7*n_sensors:8*n_sensors]
#     }
    
#     # Initialize action and communication signal
#     action = np.zeros(2)  # Horizontal and vertical thrust
#     communication_signal = 0  # CCS signal value

#     # Calculate dynamic thresholds based on environmental feedback
#     poison_distances = sensor_data['poison_dist'][sensor_data['poison_dist'] < sensor_range]
#     pursuer_distances = sensor_data['pursuer_dist'][sensor_data['pursuer_dist'] < sensor_range]
#     average_poison_dist = np.mean(poison_distances) if poison_distances.size > 0 else float('inf')
#     average_pursuer_dist = np.mean(pursuer_distances) if pursuer_distances.size > 0 else float('inf')

#     print(f"Average poison distance: {average_poison_dist}, Average pursuer distance: {average_pursuer_dist}")


#     # Adjust thresholds dynamically based on average distances
#     dynamic_thresholds = {
#         'high_urgency': 0.75 if np.isinf(average_poison_dist) else max(0.75, 1 - average_poison_dist/sensor_range),
#         'medium_urgency': 0.5 if np.isinf(average_pursuer_dist) else max(0.5, 1 - average_pursuer_dist/sensor_range)
#     }

#     # Process communication signals
#     signal_strength = np.mean(observation[-n_agents:])
#     signal_type = 'neutral'
#     for key, value in dynamic_thresholds.items():
#         if signal_strength >= value:
#             signal_type = key
#             break

#     # Strategy based on signal type and sensor data
#     if 'urgency' in signal_type:
#         # Handle urgency signals
#         if 'high' in signal_type:
#             print('high urgency signal')
#             closest_poison_index = np.argmin(sensor_data['poison_dist'])
#             action = -np.array([np.cos(closest_poison_index * 2 * np.pi / n_sensors),
#                                 np.sin(closest_poison_index * 2 * np.pi / n_sensors)])
#             communication_signal = -1
#         elif 'medium' in signal_type:
#             print('medium urgency signal')
#             closest_pursuer_index = np.argmin(sensor_data['pursuer_dist'])
#             action = -np.array([np.cos(closest_pursuer_index * 2 * np.pi / n_sensors),
#                                 np.sin(closest_pursuer_index * 2 * np.pi / n_sensors)])
#             communication_signal = -0.5
#     elif signal_type == 'neutral':
#         # No significant external signals; default to exploratory behavior
#         random_direction = np.random.rand() * 2 * np.pi
#         action = np.array([np.cos(random_direction), np.sin(random_direction)])
#         communication_signal = 0  # Maintain neutral signal
        
    

#     return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)
