import numpy as np
import csv
import time
import sys
import asyncio
import pykos
from typing import Dict, List
import os


def load_csv_data(filename, scale_factor=2.0):
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header row
        print(f"\nCSV Header: {header}")

        # Print first row for debugging
        first_row = next(csv_reader)
        values = [float(x) for x in first_row]
        print(f"\nFirst row raw values: {values}")
        action_values = [x * scale_factor for x in values[2:]]  # Skip step and env
        print(f"First row action values after scaling: {action_values}")
        action_values[8] = -action_values[8]  # Invert right knee
        data.append(action_values)

        # Load the rest of the data
        for row in csv_reader:
            values = [float(x) for x in row]
            action_values = [x * scale_factor for x in values[2:]]
            action_values[8] = -action_values[8]  # Invert right knee
            data.append(action_values)

    return np.array(data)


async def play_trajectory(kos: pykos.KOS, trajectory_data: np.ndarray, actuator_mapping: Dict[str, int]) -> None:
    print(f"Playing trajectory with {len(trajectory_data)} timesteps")
    print(f"Trajectory data shape: {trajectory_data.shape}")

    # Create a mapping from actuator IDs to CSV column indices
    id_to_column = {
        31: 0,  # left_hip_pitch_04 -> column 0
        32: 1,  # left_hip_roll_03 -> column 1
        33: 2,  # left_hip_yaw_03 -> column 2
        34: 3,  # left_knee_04 -> column 3
        35: 4,  # left_ankle_02 -> column 4
        41: 5,  # right_hip_pitch_04 -> column 5
        42: 6,  # right_hip_roll_03 -> column 6
        43: 7,  # right_hip_yaw_03 -> column 7
        44: 8,  # right_knee_04 -> column 8
        45: 9,  # right_ankle_02 -> column 9
    }

    # Create reverse mapping for logging
    id_to_name = {v: k for k, v in actuator_mapping.items()}

    # Print the first step's raw data for debugging
    if len(trajectory_data) > 0:
        print("\nFirst step raw data:")
        print(trajectory_data[0])

    try:
        for step_idx, step in enumerate(trajectory_data):
            # Create command list for this timestep
            commands = []
            print(f"\nStep {step_idx} commands:")
            print(f"Step data length: {len(step)}")
            for joint_name, motor_id in actuator_mapping.items():
                # Use the column mapping to get the correct CSV column
                column_idx = id_to_column[motor_id]
                try:
                    position = step[column_idx]
                    commands.append({"actuator_id": motor_id, "position": position})
                    print(f"  {joint_name} (motor {motor_id}, col {column_idx}): {position:.3f}")
                except IndexError:
                    print(f"  ERROR: Could not get data for {joint_name} (motor {motor_id}, col {column_idx})")

            # Send commands to robot
            await kos.actuator.command_actuators(commands)

            # Small delay to maintain control frequency
            await asyncio.sleep(0.001)  # Adjust this value based on your needs

    except KeyboardInterrupt:
        print("\nStopping trajectory playback")


async def main():
    # Define actuator mapping (motor IDs)
    actuator_mapping = {
        "left_hip_pitch_04": 31,  # Strong motors
        "left_hip_roll_03": 32,  # Medium motors
        "left_hip_yaw_03": 33,  # Medium motors
        "left_knee_04": 34,  # Strong motors
        "left_ankle_02": 35,  # Weak motors
        "right_hip_pitch_04": 41,  # Strong motors
        "right_hip_roll_03": 42,  # Medium motors
        "right_hip_yaw_03": 43,  # Medium motors
        "right_knee_04": 44,  # Strong motors
        "right_ankle_02": 45,  # Weak motors
    }

    # Load CSV data with scaling
    scale_factor = 10.0  # Adjust this value to increase/decrease action magnitude
    csv_path = "play_policy_2025-02-16_21-58-16/policy_outputs.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file at {csv_path}")
        sys.exit(1)

    try:
        trajectory_data = load_csv_data(csv_path, scale_factor)
        print(f"Actions scaled by factor of {scale_factor}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    print(f"Loaded trajectory with {len(trajectory_data)} timesteps")

    # Connect to KOS
    async with pykos.KOS() as kos:
        # Enable all motors with appropriate gains
        print("Enabling motors...")
        for motor_id in actuator_mapping.values():
            # Set gains based on motor type
            if motor_id in [31, 34, 41, 44]:  # Strong motors (R04)
                kp, kd, max_torque = 250, 5, 80
            elif motor_id in [32, 33, 42, 43]:  # Medium motors (R03)
                kp, kd, max_torque = 150, 5, 60
            else:  # Weak motors (R02)
                kp, kd, max_torque = 40, 5, 17

            await kos.actuator.configure_actuator(
                actuator_id=motor_id, kp=kp, kd=kd, max_torque=max_torque, torque_enabled=True
            )

        print("Motors enabled, starting trajectory playback...")
        await play_trajectory(kos, trajectory_data, actuator_mapping)
        print("Finished trajectory playback")


if __name__ == "__main__":
    asyncio.run(main())
