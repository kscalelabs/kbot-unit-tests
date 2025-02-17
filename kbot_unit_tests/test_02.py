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
        data.append(action_values)

        # Load the rest of the data
        for row in csv_reader:
            values = [float(x) for x in row]
            action_values = [x * scale_factor for x in values[2:]]
            data.append(action_values)

    return np.array(data)


async def play_trajectory(kos: pykos.KOS, trajectory_data: np.ndarray, actuator_mapping: Dict[str, int]) -> None:
    print(f"Playing trajectory with {len(trajectory_data)} timesteps")
    print(f"Trajectory data shape: {trajectory_data.shape}")
    print("\nPress Enter to step through each timestep, or '1' + Enter to play continuously")

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
        # Get initial input for continuous or step mode
        user_input = input()
        continuous_mode = user_input.strip() == "1"

        for step_idx, step in enumerate(trajectory_data):
            # Create command list for this timestep
            commands = []
            print(f"\nStep {step_idx} commands:")
            print(f"Step data length: {len(step)}")
            for joint_name, motor_id in actuator_mapping.items():
                # Skip arm motors during trajectory playback since they don't have CSV data
                if motor_id in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
                    continue

                # Use the column mapping to get the correct CSV column
                column_idx = id_to_column[motor_id]
                try:
                    position = step[column_idx]
                    # Negate only left knee values
                    if motor_id == 34:  # left knee only
                        print(f"  {joint_name} (motor {motor_id}, col {column_idx}): BEFORE NEGATION {position:.3f}")
                        position = position * -1.0
                        print(f"  {joint_name} (motor {motor_id}, col {column_idx}): AFTER NEGATION {position:.3f}")
                    else:
                        print(f"  {joint_name} (motor {motor_id}, col {column_idx}): {position:.3f}")
                    commands.append({"actuator_id": motor_id, "position": position})
                except IndexError:
                    print(f"  ERROR: Could not get data for {joint_name} (motor {motor_id}, col {column_idx})")

            # Send commands to robot
            await kos.actuator.command_actuators(commands)

            # If not in continuous mode, wait for user input
            if not continuous_mode:
                input("Press Enter for next step...")

            # Small delay to maintain control frequency
            await asyncio.sleep(0.001)  # Adjust this value based on your needs

    except KeyboardInterrupt:
        print("\nStopping trajectory playback")


async def main():
    # Define actuator mapping (motor IDs)
    actuator_mapping = {
        # Legs
        "left_hip_pitch_04": 31,  # Strong motors
        "left_hip_roll_03": 32,  # Medium motors
        "left_hip_yaw_03": 33,  # Medium motors
        "left_knee_04": 34,  # Strong motors
        "left_ankle_02": 35,  # Weak motors
        "right_hip_pitch_04": 41,  # Strong motors
        "right_hip_roll_03": 42,  # Medium motors
        "right_hip_yaw_03": 43,  # Medium motors
        "right_knee_04": 44,  # Strong motors
        "right_ankle_02": 45,  # Weak motors,
        # Arms
        "left_shoulder_pitch": 11,
        "left_shoulder_roll": 12,
        "left_shoulder_yaw": 13,
        "left_elbow": 14,
        "left_wrist": 15,
        "right_shoulder_pitch": 21,
        "right_shoulder_roll": 22,
        "right_shoulder_yaw": 23,
        "right_elbow": 24,
        "right_wrist": 25,
    }

    # Load CSV data with scaling
    scale_factor = 7.0  # Adjust this value to increase/decrease action magnitude
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
            if motor_id in [31, 41]:  # RS04_Pitch (hip pitch)
                kp, kd, max_torque = 85, 3, 80
            elif motor_id in [32, 42]:  # RS03_Roll (hip roll)
                kp, kd, max_torque = 85, 2, 60
            elif motor_id in [33, 43]:  # RS03_Yaw (hip yaw)
                kp, kd, max_torque = 30, 2, 60
            elif motor_id in [34, 44]:  # RS04_Knee (knee)
                kp, kd, max_torque = 60, 2, 80
            elif motor_id in [35, 45]:  # RS02 (ankle)
                kp, kd, max_torque = 80, 1, 17
            else:  # Arm motors - set to zero position with low gains
                kp, kd, max_torque = 20, 2, 20

            await kos.actuator.configure_actuator(
                actuator_id=motor_id, kp=kp, kd=kd, max_torque=max_torque, torque_enabled=True
            )

            # Set arm motors to zero position
            if motor_id in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
                await kos.actuator.command_actuators([{"actuator_id": motor_id, "position": 0.0}])

        print("Motors enabled, starting trajectory playback...")
        await play_trajectory(kos, trajectory_data, actuator_mapping)
        print("Finished trajectory playback")


if __name__ == "__main__":
    asyncio.run(main())
