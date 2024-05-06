# heuristic_signal_policy.py

import numpy as np

def scale_action(action, distance):
    """Scale the action vector based on the inverse of the distance to increase urgency."""
    return action * (1 + distance)  # Increase impact as distance decreases


def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents, agent_id):
    indices = np.arange(n_sensors) * 2 * np.pi / n_sensors
    action = np.zeros(2)
    communication_signal = 0
    sensor_range = 1 # no matter the env sensor range, the actual distance range is always scaled down to [0,1]

    # Sensor data extraction
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    food_speed = observation[3 * n_sensors:4 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]
    poison_speed = observation[5 * n_sensors:6 * n_sensors]
    pursuer_dist = observation[6 * n_sensors:7 * n_sensors]
    communication_signals = observation[-(n_agents - 1):]  # Exclude own signal

    # Calculate risk values for food and poison
    food_risk_values = food_dist + 0.5 * np.abs(food_speed)
    poison_risk_values = poison_dist + 0.5 * np.abs(poison_speed)

    # Immediate poison avoidance logic
    if np.any(poison_dist < 0.2) and np.min(poison_risk_values) < np.min(food_risk_values):
        closest_poison_index = np.argmin(poison_risk_values)
        direction = np.array([np.cos(indices[closest_poison_index]), np.sin(indices[closest_poison_index])])
        action = -direction  # Move away from poison
        action = scale_action(action, poison_dist[closest_poison_index])  # Scale based on distance
        communication_signal = -1  # Indicate danger


    # Pairing and food collection logic, using communication signals to coordinate
    elif np.any(food_dist < sensor_range):
        # Check if another agent is signaling readiness to collect food
        for i, signal in enumerate(communication_signals):
            if signal > 0 and food_dist[i] < sensor_range:
                closest_food_index = i
                direction = np.array([np.cos(indices[closest_food_index]), np.sin(indices[closest_food_index])])
                action = direction  # Move towards food
                action = scale_action(action, food_dist[closest_food_index])  # Scale based on distance
                communicication_signal = 1  # Emit signal to coordinate food collection
                communication_signal = 1  # Emit signal to coordinate food collection
                break

    # Maintain pairing or random movement if no immediate tasks
    if np.linalg.norm(action) == 0:
        if np.any(pursuer_dist < sensor_range):
            closest_pursuer_index = np.argmin(pursuer_dist)
            action = np.array([np.cos(indices[closest_pursuer_index]), np.sin(indices[closest_pursuer_index])])
            action = scale_action(action, pursuer_dist[closest_pursuer_index])  # Scale based on proximity
            communication_signal = 0.5  # Maintain pair signal
        else:
            random_direction = np.random.rand() * 2 * np.pi
            action = np.array([np.cos(random_direction), np.sin(random_direction)])
            action = scale_action(action, 1)  # Random move, no specific target
    action = np.clip(action, -1, 1)
    return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)


# # Pairing and food collection logic
# elif np.any(pursuer_dist < sensor_range):
#     closest_food_index = np.argmin(food_risk_values)
#     if food_dist[closest_food_index] < sensor_range:
#         direction = np.array([np.cos(indices[closest_food_index]), np.sin(indices[closest_food_index])])
#         action = direction  # Move towards food
#         action = scale_action(action, food_dist[closest_food_index])  # Scale based on distance
#         communication_signal = 0.5  # Signal to collect food
            
# def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents, agent_id):
#     # Sensor and agent indices calculation
#     indices = np.arange(n_sensors) * 2 * np.pi / n_sensors
#     action = np.zeros(2)
#     communication_signal = 0
#     sensor_range = 1  # no matter the env sensor range, the actual distance range is always scaled down to [0,1]

#     # Sensor data extraction
#     food_dist = observation[2 * n_sensors:3 * n_sensors]
#     food_speed = observation[3 * n_sensors:4 * n_sensors]
#     poison_dist = observation[4 * n_sensors:5 * n_sensors]
#     poison_speed = observation[5 * n_sensors:6 * n_sensors]
#     pursuer_dist = observation[6 * n_sensors:7 * n_sensors]

#     # Extracting communication signals from other agents
#     # Assuming the signal from each agent is appended after environmental data
#     communication_signals = observation[-(n_agents - 1):]  # Exclude own signal

#     # Calculate risk values for food and poison
#     food_risk_values = food_dist + 0.5 * np.abs(food_speed)
#     poison_risk_values = poison_dist + 0.5 * np.abs(poison_speed)

#     # Immediate poison avoidance logic
#     if np.any(poison_dist < 0.2) and np.min(poison_risk_values) < np.min(food_risk_values):
#         closest_poison_index = np.argmin(poison_risk_values)
#         direction = np.array([np.cos(indices[closest_poison_index]), np.sin(indices[closest_poison_index])])
#         action = -direction  # Move away from poison
#         action = scale_action(action, poison_dist[closest_poison_index])  # Scale based on distance
#         communication_signal = -1  # Indicate danger

#     # Pairing and food collection logic, using communication signals to coordinate
#     elif np.any(food_dist < sensor_range):
#         # Check if another agent is signaling readiness to collect food
#         for i, signal in enumerate(communication_signals):
#             if signal > 0 and food_dist[i] < sensor_range:
#                 closest_food_index = i
#                 direction = np.array([np.cos(indices[closest_food_index]), np.sin(indices[closest_food_index])])
#                 action = direction  # Move towards food
#                 action = scale_action(action, food_dist[closest_food_index])  # Scale based on distance
#                 communication_signal = 1  # Emit signal to coordinate food collection
#                 break

#     # Default behavior if no specific action is determined
#     if np.linalg.norm(action) == 0:
#         random_direction = np.random.rand() * 2 * np.pi
#         action = np.array([np.cos(random_direction), np.sin(random_direction)])
#         action = scale_action(action, 1)  # Random move, no specific target
#         communication_signal = 0  # Default signal indicating no immediate objective

#     action = np.clip(action, -1, 1)
#     return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)




# import numpy as np

# def scale_action(action, distance):
#     return action * (1+ distance)



# def communication_heuristic_policy(observation, n_sensors, sensor_range, n_agents, agent_id):
 

#     indices = np.arange(n_sensors) * 2 * np.pi / n_sensors
#     action = np.zeros(2)
#     communication_signal = 0
    
#     sensor_range = 1 # no matter the env sensor range, the actual distance range is always scaled down to [0,1]
#     poison_threshold = 0.2
    

#     # Extract sensor data
#     food_dist = observation[2 * n_sensors:3 * n_sensors]
#     food_speed = observation[3 * n_sensors:4 * n_sensors]
#     poison_dist = observation[4 * n_sensors:5 * n_sensors]
#     poison_speed = observation[5 * n_sensors:6 * n_sensors]
#     pursuer_dist = observation[6 * n_sensors:7 * n_sensors]
#     communication_signals = observation[-(n_agents - 1):]  # Exclude own signal

#     food_risk_values = food_dist + 0.5 * np.abs(food_speed)
#     poison_risk_values = poison_dist + 0.5 * np.abs(poison_speed)
    
    
#     other_agents = [i for i in range(1, n_agents+1) if i != agent_id]
#     ods = other_agents
    
#     # Additional agent proximity and pairing logic
#     list_of_close_pursuers_indices = np.where((pursuer_dist < sensor_range))[0]
    
   
#     for i, dist in enumerate(list_of_close_pursuers_indices):
#         if (i+1) in other_agents and (i+1) != agent_id:
#             other_agents[i] = pursuer_dist[dist]
#             break
    
    
#     closest_pursuer_index = np.argmin(other_agents)
#     is_pursuer_close = 0 < pursuer_dist[closest_pursuer_index] < sensor_range
    

#     # Immediate poison avoidance
#     if np.any((poison_dist < poison_threshold)) and np.min(poison_risk_values) < np.min(food_risk_values): # keep poison a priority at lower distance
#         closest_poison_index = np.argmin(poison_risk_values)
#         action = -np.array([np.cos(indices[closest_poison_index]), np.sin(indices[closest_poison_index])])
#         communication_signal = -1
#     elif is_pursuer_close:
#         # Pairing logic
#         if communication_signals[closest_pursuer_index] > 0:  # Positive signal implies an intent to pair
#             communication_signal = communication_signals[closest_pursuer_index]  # Echo the pairing signal
#             if np.any((food_dist < sensor_range)):
#                 closest_food_index = np.argmin(food_risk_values)
#                 if food_dist[closest_food_index] < sensor_range:
#                     action = np.array([np.cos(indices[closest_food_index]), np.sin(indices[closest_food_index])])
#                     communication_signal = 0.5  # Signal to collect food
#         else:
#             # Send a pairing signal
#             communication_signal = 0.5

#     # Maintain pair
#     if is_pursuer_close and np.all(action == 0):
#         action = np.array([np.cos(indices[closest_pursuer_index]), np.sin(indices[closest_pursuer_index])])
#         communication_signal = 0.5  # Maintain pair signal

#     other_agents = np.array(other_agents)
    
#     # Agent identifiers sorted by proximity
#     proximal_agent_indices = np.where((other_agents != np.any(ods)) & (other_agents < sensor_range) & (communication_signals != 0))[0]
#     if len(proximal_agent_indices) > 0:
#         for idx in proximal_agent_indices:
#             if other_agents[idx] < sensor_range:
#                 # Attempt to pair with the closest agent not already paired
#                 if communication_signals[idx] == 0: 
#                     communication_signal = agent_id  
#                     action = np.array([np.cos(indices[idx]), np.sin(indices[idx])])
#                     break

#     # Random movement if no immediate tasks
#     if np.all(action == 0):
#         random_direction = np.random.rand() * 2 * np.pi
#         action = np.array([np.cos(random_direction), np.sin(random_direction)])

    

#     return np.concatenate((action, [communication_signal]), axis=None).astype(np.float32)

