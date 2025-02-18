import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_armature(file_path: str, time_start: float = 1, time_shutoff: float = 1.5) -> None:
    # Read the JSON file
    with open(file_path, 'r') as f:
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
        
        velocity_threshold = 0.01

        # Evaluate the expected constant velocity on transient up and down conditions
        # Filter data points before time_shutoff and with positive velocity
        transient_df_up = sim_actuator[
            (sim_actuator['time'] < time_shutoff) &
            (sim_actuator['time'] > time_start)
            #(sim_actuator['velocity'] > velocity_threshold)
        ].copy()

        # Filter data points after time_shutoff and with negative velocity
        transient_df_down = sim_actuator[
            (sim_actuator['time'] > time_shutoff) & 
            (sim_actuator['velocity'] > velocity_threshold)
        ].copy()
        
        # Calculate acceleration for transient up phase
        if not transient_df_up.empty:
            # Calculate time differences between consecutive points
            transient_df_up['time_diff'] = transient_df_up['time'].diff()
            
            # Calculate velocity differences between consecutive points
            transient_df_up['velocity_diff'] = transient_df_up['velocity'].diff()
            
            # Calculate discrete acceleration (dv/dt)
            transient_df_up['acceleration'] = transient_df_up['velocity_diff'] / transient_df_up['time_diff']
            
            # Calculate mean acceleration (excluding first row which has NaN from diff)
            #mean_acceleration_up = transient_df_up['acceleration'].mean(skipna=True)
            
            # approximate acceleration::
            start_time = transient_df_up['time'].iloc[0]
            end_time = transient_df_up['time'].iloc[-1]
            start_velocity = transient_df_up.iloc[0, transient_df_up.columns.get_loc('velocity')]
            end_velocity = transient_df_up.iloc[-1, transient_df_up.columns.get_loc('velocity')]
            mean_acceleration_up = (end_velocity - start_velocity) / (end_time - start_time)
            
            #print(f"Actuator {actuator_id} - Mean acceleration during transient up: {mean_acceleration:.2f} rad/s²")

        # Calculate acceleration for transient down phase
        if not transient_df_down.empty:
            # Calculate time differences between consecutive points
            transient_df_down['time_diff'] = transient_df_down['time'].diff()
            
            # Calculate velocity differences between consecutive points
            transient_df_down['velocity_diff'] = transient_df_down['velocity'].diff()
            
            # Calculate discrete acceleration (dv/dt)
            transient_df_down['acceleration'] = transient_df_down['velocity_diff'] / transient_df_down['time_diff']
            
            # Calculate mean acceleration (excluding first row which has NaN from diff)
            #mean_acceleration_down = transient_df_down['acceleration'].mean(skipna=True)
            
            # approximate acceleration::
            start_time = transient_df_down['time'].iloc[0]
            end_time = transient_df_down['time'].iloc[-1]
            start_velocity = transient_df_down.iloc[0, transient_df_down.columns.get_loc('velocity')]
            end_velocity = transient_df_down.iloc[-1, transient_df_down.columns.get_loc('velocity')]
            mean_acceleration_down = (end_velocity - start_velocity) / (end_time - start_time)
            
            #print(f"Actuator {actuator_id} - Mean acceleration during transient down: {mean_acceleration:.2f} rad/s²")

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
        # Add acceleration line for transient up phase if data exists
        if not transient_df_up.empty:
            # Get start time and velocity from first point in transient
            t0 = transient_df_up['time'].iloc[0]
            v0 = transient_df_up['velocity'].iloc[0]
            
            # Create line with calculated acceleration
            t_line = np.array([t0, sim_actuator['time'].max()])
            v_line = v0 + mean_acceleration_up * (t_line - t0)
            
            axes[i,1].plot(t_line, v_line, 'r--', 
                          label=f'Acceleration={mean_acceleration_up:.2f} rad/s²')
            axes[i,1].legend()

        # Add acceleration line for transient down phase if data exists
        if not transient_df_down.empty:
            # Get start time and velocity from first point in transient
            t0 = transient_df_down['time'].iloc[0]
            v0 = transient_df_down['velocity'].iloc[0]
            
            # Create line with calculated acceleration
            t_line = np.array([t0, sim_actuator['time'].max()])
            v_line = v0 + mean_acceleration_down * (t_line - t0)
            
            axes[i,1].plot(t_line, v_line, 'b--',
                          label=f'Acceleration={mean_acceleration_down:.2f} rad/s²')
            axes[i,1].legend()

        # Plot torque data
        # Filter torque data for transient up phase
        if not transient_df_up.empty:
            mean_commanded_torque_up = transient_df_up['commanded_torque'].mean()
            axes[i,2].axhline(y=mean_commanded_torque_up, color='r', linestyle=':',
                            label=f'Mean Torque Up={mean_commanded_torque_up:.2f} Nm')
        
        

        # Calculate the mean torque during the transient up and down phases
        axes[i,2].plot(sim_actuator['time'], sim_actuator['commanded_torque'],
                    label='Commanded', linestyle='--')
        axes[i,2].plot(sim_actuator['time'], sim_actuator['torque'],
                    label='Actual')
        axes[i,2].set_xlabel('Time (s)')
        axes[i,2].set_ylabel('Torque')
        axes[i,2].set_title(f'Actuator {actuator_id} - Torque')
        axes[i,2].legend()
        axes[i,2].grid(True)

        print(f"Actuator {actuator_id} - Calc Armature of {mean_commanded_torque_up / (mean_acceleration_up + mean_acceleration_down)} kg-m^2")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def __main__(file_path: str = "./utils/test07_sim_actuator_states.json"):
    plot_armature(file_path=file_path)

if __name__ == "__main__":
    __main__()