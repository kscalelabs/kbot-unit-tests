#TODO: (1) Send trajectory plan to kos-sim and sim. (2) Trajectory comes from motion planning primitive
#TODO: (3) Plot the trajectory (4) Send trajectory to real robot (5) Trajectory is a squating motion

#* ONE move from zero position to the squat starting position (start from on ground)
#* TWO scripted motion of moving through the squat positions


import argparse
import asyncio
import logging
import time
from pathlib import Path
from dataclasses import dataclass
import colorlogging
from pykos import KOS

from motion_planning_primitive import find_points_to_target

logger = logging.getLogger(__name__)

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    flip_sign: bool = False


ACTUATOR_LIST = [
    # actuator id, nn id, kp, kd, max_torque, flip_sign
    Actuator(11, 1, 150.0, 8.0, 60.0, True),  # left_shoulder_pitch_03
    Actuator(12, 5, 150.0, 8.0, 60.0, False),  # left_shoulder_roll_03
    Actuator(13, 9, 50.0, 5.0, 17.0, False),  # left_shoulder_yaw_02
    Actuator(14, 13, 50.0, 5.0, 17.0, False),  # left_elbow_02
    Actuator(15, 17, 20.0, 2.0, 17.0, False),  # left_wrist_02
    Actuator(21, 3, 150.0, 8.0, 60.0, False),  # right_shoulder_pitch_03
    Actuator(22, 7, 150.0, 8.0, 60.0, True),  # right_shoulder_roll_03
    Actuator(23, 11, 50.0, 2.0, 17.0, True),  # right_shoulder_yaw_02
    Actuator(24, 15, 50.0, 5.0, 17.0, True),  # right_elbow_02
    Actuator(25, 19, 20.0, 2.0, 17.0, False),  # right_wrist_02
    Actuator(31, 0, 100.0, 6.1504, 80.0, True),  # left_hip_pitch_04 (RS04_Pitch)
    Actuator(32, 4, 50.0, 11.152, 60.0, False),  # left_hip_roll_03 (RS03_Roll) #* DONE
    Actuator(33, 8, 50.0, 11.152, 60.0, False),  # left_hip_yaw_03 (RS03_Yaw)
    Actuator(34, 12, 100.0, 6.1504, 80.0, True),  # left_knee_04 (RS04_Knee)
    Actuator(35, 16, 20.0, 0.6, 17.0, False),  # left_ankle_02 (RS02)
    Actuator(41, 2, 100, 7.0, 80.0, False),  # right_hip_pitch_04 (RS04_Pitch) #* DONE
    Actuator(42, 6, 50.0, 11.152, 60.0, True),  # right_hip_roll_03 (RS03_Roll)
    Actuator(43, 10, 50.0, 11.152, 60.0, True),  # right_hip_yaw_03 (RS03_Yaw)
    Actuator(44, 14, 100.0, 6.1504, 80.0, False),  # right_knee_04 (RS04_Knee) #* DONE
    Actuator(45, 18, 20.0, 0.6, 17.0, True),  # right_ankle_02 (RS02)
]


async def move_actuators_with_trajectory(sim_kos: KOS, actuator_ids: list[int], target_angles: list[float], 
                                       real_kos: KOS = None) -> None:
    """
    Move multiple actuators from their current positions to target positions using trajectories.
    
    Args:
        sim_kos: KOS simulation instance
        actuator_ids: List of IDs of the actuators to move
        target_angles: List of target angles in degrees (must match length of actuator_ids)
        real_kos: Optional KOS real robot instance
    """
    if len(actuator_ids) != len(target_angles):
        raise ValueError("Number of actuator IDs must match number of target angles")
    
    # Create a mapping of actuator_id to flip_sign from ACTUATOR_LIST
    flip_sign_map = {actuator.actuator_id: actuator.flip_sign for actuator in ACTUATOR_LIST}
    
    # Apply sign flipping to target angles if needed
    adjusted_target_angles = []
    for actuator_id, target_angle in zip(actuator_ids, target_angles):
        # Check if this actuator needs sign flipping
        if actuator_id in flip_sign_map and flip_sign_map[actuator_id]:
            adjusted_target_angles.append(-target_angle)  # Flip the sign
        else:
            adjusted_target_angles.append(target_angle)  # Keep original
    
    # Get the current positions of all actuators
    state_response = await sim_kos.actuator.get_actuators_state(actuator_ids)
    current_angles = [state.position for state in state_response.states]
    
    # for i, (actuator_id, current_angle, target_angle) in enumerate(zip(actuator_ids, current_angles, adjusted_target_angles)):
        # logger.info(f"Current position of actuator {actuator_id}: {current_angle} degrees, target: {target_angle} degrees")
    
    # Generate trajectories for each actuator
    trajectories = []
    max_trajectory_length = 0
    max_trajectory_time = 0
    
    for i, (actuator_id, current_angle, target_angle) in enumerate(zip(actuator_ids, current_angles, adjusted_target_angles)):
        logger.info(f"Generating trajectory for actuator {actuator_id} from {current_angle} to {target_angle} degrees...")
        angles, velocities, times = find_points_to_target(
            current_angle=current_angle,
            target=target_angle,
        )
        trajectories.append((angles, velocities, times))
        max_trajectory_length = max(max_trajectory_length, len(angles))
        max_trajectory_time = max(max_trajectory_time, times[-1])
    
    # Execute the trajectories
    logger.info(f"Executing trajectories for {len(actuator_ids)} actuators...")
    start_time = time.time()
    
    # Create a time grid for all trajectories
    dt = 1.0 / 100.0  # Assuming 100Hz update rate
    num_steps = int(max_trajectory_time / dt) + 1
    time_grid = [i * dt for i in range(num_steps)]
    
    for step, t in enumerate(time_grid):
        current_time = time.time()
        
        # Create commands for all actuators at this time step
        command = []
        for i, (actuator_id, (angles, velocities, times)) in enumerate(zip(actuator_ids, trajectories)):
            # Find the closest time point in this trajectory
            if t <= times[-1]:
                # Find the index of the closest time point
                idx = min(range(len(times)), key=lambda j: abs(times[j] - t))
                
                # Get position and velocity at this time
                position = angles[idx]
                velocity = velocities[idx-1] if idx > 0 and idx-1 < len(velocities) else 0.0
                
                # Add command for this actuator
                command.append({
                    "actuator_id": actuator_id,
                    "position": position,
                    "velocity": velocity
                })
            else:
                # If we're past the end of this trajectory, use the final position
                command.append({
                    "actuator_id": actuator_id,
                    "position": angles[-1],
                    "velocity": 0.0
                })
        
        # Send commands to sim and optionally real robot
        command_tasks = [sim_kos.actuator.command_actuators(command)]
        if real_kos:
            command_tasks.append(real_kos.actuator.command_actuators(command))
        
        await asyncio.gather(*command_tasks)
        
        # Calculate time to sleep to maintain update rate
        next_time = start_time + t + dt
        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)
    
    # Ensure we reach the final positions
    final_command = [
        {
            "actuator_id": actuator_id,
            "position": target_angle,
            "velocity": 0.0
        }
        for actuator_id, target_angle in zip(actuator_ids, adjusted_target_angles)
    ]
    
    command_tasks = [sim_kos.actuator.command_actuators(final_command)]
    if real_kos:
        command_tasks.append(real_kos.actuator.command_actuators(final_command))
    
    await asyncio.gather(*command_tasks)
    
    # Get the final positions to verify
    state_response = await sim_kos.actuator.get_actuators_state(actuator_ids)
    for i, (actuator_id, state) in enumerate(zip(actuator_ids, state_response.states)):
        logger.info(f"Final position of actuator {actuator_id}: {state.position} degrees")


async def setup_actuator(sim_kos: KOS, actuator_id: int, kp: float, kd: float, 
                        max_torque: float = None, real_kos: KOS = None) -> None:
    """
    Setup an actuator by configuring its gains and enabling it.
    
    Args:
        sim_kos: KOS simulation instance
        actuator_id: ID of the actuator to setup
        kp: Position gain
        kd: Velocity gain
        max_torque: Maximum torque (only used for real robot)
        real_kos: Optional KOS real robot instance
    """
    logger.info(f"Configuring actuator {actuator_id}...")
    
    # First disable the actuator
    await sim_kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=False)
    if real_kos:
        await real_kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=False)
    
    await asyncio.sleep(1)
    
    # Configure the actuator with appropriate gains
    config_commands = [
        sim_kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=kp,
            kd=kd,
            torque_enabled=True,
        )
    ]
    if real_kos:
        config_commands.append(
            real_kos.actuator.configure_actuator(
                actuator_id=actuator_id,
                kp=kp,
                kd=kd,
                max_torque=max_torque if max_torque else 80.0,
                torque_enabled=True,
            )
        )
    await asyncio.gather(*config_commands)

async def disable_actuator(sim_kos: KOS, actuator_id: int, real_kos: KOS = None) -> None:
    """
    Disable an actuator.
    
    Args:
        sim_kos: KOS simulation instance
        actuator_id: ID of the actuator to disable
        real_kos: Optional KOS real robot instance
    """
    logger.info(f"Disabling actuator {actuator_id}...")
    disable_commands = [
        sim_kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=False)
    ]
    if real_kos:
        disable_commands.append(
            real_kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=False)
        )
    await asyncio.gather(*disable_commands)

async def demo_multi_actuator_movement(sim_kos: KOS, real_kos: KOS = None) -> None:
    """
    Demonstrate moving multiple actuators simultaneously.
    """
    # Reset the simulation
    await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
    
    # Define actuator IDs for right and left knee
    right_knee_id = 44  # right_knee_04 (RS04_Knee)
    left_knee_id = 34   # left_knee_04 (LS04_Knee)
    
    # Setup both knee actuators
    for actuator_id in [right_knee_id, left_knee_id]:
        await setup_actuator(
            sim_kos=sim_kos,
            actuator_id=actuator_id,
            kp=100.0,
            kd=6.1504,
            max_torque=80.0,
            real_kos=real_kos
        )
    
    # Move both knees to -30 degrees simultaneously
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=[right_knee_id, left_knee_id],
        target_angles=[-30.0, -30.0],
        real_kos=real_kos
    )
    
    # Wait a moment at the target position
    await asyncio.sleep(1)
    
    # Move knees to different positions simultaneously
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=[right_knee_id, left_knee_id],
        target_angles=[-15.0, -45.0],  # Right knee to -15, left knee to -45
        real_kos=real_kos
    )
    
    # Wait a moment at the target position
    await asyncio.sleep(1)
    
    # Move both knees back to 0 degrees simultaneously
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=[right_knee_id, left_knee_id],
        target_angles=[0.0, 0.0],
        real_kos=real_kos
    )
    
    # Disable the actuators
    for actuator_id in [right_knee_id, left_knee_id]:
        await disable_actuator(sim_kos, actuator_id, real_kos)

async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--deploy", action="store_true",
                       help="Connect to the real robot (default: simulation only)")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    if not args.deploy:
        async with KOS(ip=args.host, port=args.port) as sim_kos:
            await demo_multi_actuator_movement(sim_kos)
    else:
        async with KOS(ip=args.host, port=args.port) as sim_kos, \
                   KOS(ip="100.117.248.15", port=args.port) as real_kos:
            await demo_multi_actuator_movement(sim_kos, real_kos)


if __name__ == "__main__":
    asyncio.run(main())




