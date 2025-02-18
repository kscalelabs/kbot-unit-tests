import json
import matplotlib.pyplot as plt
import pandas as pd



# Read the JSON file
with open('test07_sim_actuator_states.json', 'r') as f:
    sim_data = json.load(f)

# Convert to pandas DataFrame
sim_df = pd.DataFrame(sim_data)

kp = sim_df['kp'].iloc[0]
kd = sim_df['kd'].iloc[0]

# Get unique actuator IDs
actuator_ids = sim_df['actuator_id'].unique()

# Create figure with subplots for each actuator
num_actuators = len(actuator_ids)
fig, axes = plt.subplots(num_actuators, 3, figsize=(20, 4*num_actuators))
fig.suptitle(f'Commanded vs Actual Position, Velocity and Torque Over Time by Actuator (kp={kp}, kd={kd})')

# If there's only one actuator, wrap axes in a 2D array
if num_actuators == 1:
    axes = axes.reshape(1, 3)

# Plot for each actuator
for i, actuator_id in enumerate(actuator_ids):
    sim_actuator = sim_df[sim_df['actuator_id'] == actuator_id]
    
    # Plot position data
    axes[i,0].plot(sim_actuator['time'], sim_actuator['commanded_position'], 
                   label='Commanded', linestyle='--')
    axes[i,0].plot(sim_actuator['time'], sim_actuator['position'], 
                   label='Actual')
    axes[i,0].set_xlabel('Time (s)')
    axes[i,0].set_ylabel('Position')
    axes[i,0].set_title(f'Actuator {actuator_id} - Position')
    axes[i,0].legend()
    axes[i,0].grid(True)

    # Plot velocity data
    axes[i,1].plot(sim_actuator['time'], sim_actuator['commanded_velocity'],
                   label='Commanded', linestyle='--')
    axes[i,1].plot(sim_actuator['time'], sim_actuator['velocity'],
                   label='Actual')
    axes[i,1].set_xlabel('Time (s)')
    axes[i,1].set_ylabel('Velocity')
    axes[i,1].set_title(f'Actuator {actuator_id} - Velocity')
    axes[i,1].legend()
    axes[i,1].grid(True)

    # Plot torque data
    axes[i,2].plot(sim_actuator['time'], sim_actuator['commanded_torque'],
                   label='Commanded', linestyle='--')
    axes[i,2].plot(sim_actuator['time'], sim_actuator['torque'],
                   label='Actual')
    axes[i,2].set_xlabel('Time (s)')
    axes[i,2].set_ylabel('Torque')
    axes[i,2].set_title(f'Actuator {actuator_id} - Torque')
    axes[i,2].legend()
    axes[i,2].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
