"""Script for testing cable flex life by moving multiple joints with different sine curves."""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pykos import KOS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    name: str  # Added name for better logging


@dataclass
class JointMotionProfile:
    """Configuration for a joint's motion pattern."""
    amplitude: float  # degrees (half of peak-to-peak)
    offset: float  # degrees
    frequency: float  # Hz
    phase: float = 0.0  # radians, allows for phase shifting between joints
    time_offset: float = 0.0  # seconds, shifts the waveform in time without affecting start time
    waveform: str = "sine"  # "sine", "square", "triangle", or "sawtooth"


@dataclass
class StateLog:
    time: float
    actuator_id: int
    position: float
    velocity: float
    commanded_position: float
    commanded_velocity: float
    kp: float
    kd: float


class TestData:
    def __init__(self):
        self.states: List[StateLog] = []
    
    def log_state(self, time: float, actuator_id: int, position: float, velocity: float, 
                 commanded_position: float, commanded_velocity: float, kp: float, kd: float):
        self.states.append(StateLog(time, actuator_id, position, velocity, 
                                  commanded_position, commanded_velocity, kp, kd))
    
    def save_to_json(self, filename: str):
        data = [{"time": s.time, "actuator_id": s.actuator_id, 
                 "position": s.position, "velocity": s.velocity,
                 "commanded_position": s.commanded_position,
                 "commanded_velocity": s.commanded_velocity,
                 "kp": s.kp, "kd": s.kd}
                for s in self.states]
        
        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


# Define all available actuators
ACTUATOR_LIST: list[Actuator] = [
    # actuator id, nn id, kp, kd, max_torque, name
    Actuator(11, 1, 150.0, 8.0, 60.0, "left_shoulder_pitch_03"),
    Actuator(12, 5, 150.0, 8.0, 60.0, "left_shoulder_roll_03"),
    Actuator(13, 9, 50.0, 5.0, 17.0, "left_shoulder_yaw_02"),
    Actuator(14, 13, 50.0, 5.0, 17.0, "left_elbow_02"),
    Actuator(15, 17, 20.0, 2.0, 17.0, "left_wrist_02"),
    Actuator(21, 3, 150.0, 8.0, 60.0, "right_shoulder_pitch_03"),
    Actuator(22, 7, 150.0, 8.0, 60.0, "right_shoulder_roll_03"),
    Actuator(23, 11, 50.0, 5.0, 17.0, "right_shoulder_yaw_02"),
    Actuator(24, 15, 50.0, 5.0, 17.0, "right_elbow_02"),
    Actuator(25, 19, 20.0, 2.0, 17.0, "right_wrist_02"),
    Actuator(31, 0, 100.0, 6.1504, 80.0, "left_hip_pitch_04"),
    Actuator(32, 4, 50.0, 11.152, 60.0, "left_hip_roll_03"),
    Actuator(33, 8, 50.0, 11.152, 60.0, "left_hip_yaw_03"),
    Actuator(34, 12, 100.0, 6.1504, 80.0, "left_knee_04"),
    Actuator(35, 16, 20.0, 0.6, 17.0, "left_ankle_02"),
    Actuator(41, 2, 100.0, 7.0, 80.0, "right_hip_pitch_04"),
    Actuator(42, 6, 50.0, 11.152, 60.0, "right_hip_roll_03"),
    Actuator(43, 10, 50.0, 11.152, 60.0, "right_hip_yaw_03"),
    Actuator(44, 14, 100.0, 6.1504, 80.0, "right_knee_04"),
    Actuator(45, 18, 20.0, 0.6, 17.0, "right_ankle_02"),
]


def calculate_position_velocity(t: float, profile: JointMotionProfile) -> Tuple[float, float]:
    """Calculate position and velocity based on the motion profile and time."""
    angular_freq = 2 * np.pi * profile.frequency
    
    # Apply time offset to the time value
    phase_shifted_t = t + profile.phase / angular_freq + profile.time_offset
    
    if profile.waveform == "sine":
        position = profile.offset + profile.amplitude * np.sin(angular_freq * phase_shifted_t)
        velocity = profile.amplitude * angular_freq * np.cos(angular_freq * phase_shifted_t)
    
    elif profile.waveform == "square":
        position = profile.offset + profile.amplitude * np.sign(np.sin(angular_freq * phase_shifted_t))
        velocity = 0.0  # Velocity is technically infinite at transitions, so we set to 0
    
    elif profile.waveform == "sawtooth":
        # Normalized sawtooth from 0 to 1
        sawtooth = (phase_shifted_t * profile.frequency) % 1.0
        position = profile.offset + profile.amplitude * (2.0 * sawtooth - 1.0)
        velocity = profile.amplitude * 2.0 * profile.frequency  # Constant velocity except at transitions
    
    elif profile.waveform == "triangle":
        # Triangle wave using absolute value of sawtooth
        sawtooth = (phase_shifted_t * profile.frequency) % 1.0
        triangle = 1.0 - 2.0 * abs(sawtooth - 0.5)
        position = profile.offset + profile.amplitude * triangle
        
        # Triangle wave velocity alternates between positive and negative
        direction = 1.0 if sawtooth < 0.5 else -1.0
        velocity = direction * profile.amplitude * 4.0 * profile.frequency
    
    else:
        raise ValueError(f"Unknown waveform type: {profile.waveform}")
    
    return position, velocity


async def disable_motors(sim_kos: KOS, real_kos: KOS = None, actuator_ids: List[int] = None) -> None:
    """
    Safely disable motors to prevent damage when stopping the test.
    
    Args:
        sim_kos: KOS instance for simulation
        real_kos: KOS instance for real robot (optional)
        actuator_ids: List of actuator IDs to disable (if None, disables all actuators)
    """
    logger.info("Emergency stop: Disabling motors...")
    
    if actuator_ids is None:
        # If no specific actuators provided, disable all known actuators
        actuator_ids = [a.actuator_id for a in ACTUATOR_LIST]
    
    active_actuators = [a for a in ACTUATOR_LIST if a.actuator_id in actuator_ids]
    
    for actuator in active_actuators:
        disable_commands = [
            sim_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        ]
        if real_kos:
            disable_commands.append(
                real_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
            )
        try:
            await asyncio.gather(*disable_commands)
            logger.info(f"Disabled {actuator.name} (ID: {actuator.actuator_id})")
        except Exception as e:
            logger.error(f"Failed to disable {actuator.name} (ID: {actuator.actuator_id}): {e}")


async def test_client(sim_kos: KOS, real_kos: KOS = None, 
                     joint_profiles: Dict[int, JointMotionProfile] = None,
                     duration: float = 10.0) -> None:
    """
    Run the test with specified joint motion profiles.
    
    Args:
        sim_kos: KOS instance for simulation
        real_kos: KOS instance for real robot (optional)
        joint_profiles: Dictionary mapping actuator_id to JointMotionProfile
        duration: Test duration in seconds
    """
    logger.info("Starting cable flex life test...")
    
    if joint_profiles is None or len(joint_profiles) == 0:
        logger.warning("No joint profiles specified. Using default profile for right knee.")
        return
    
    # Initialize separate test data loggers for sim and real
    sim_data = TestData() if sim_kos else None
    real_data = TestData() if real_kos else None

    # Get list of actuator IDs to move
    actuator_ids_to_move = list(joint_profiles.keys())
    logger.info(f"Moving actuators: {actuator_ids_to_move}")
    
    try:
        # Reset the simulation if we're using it
        if sim_kos:
            await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})

        # First disable only the motors we need to move
        logger.info("Disabling motors for test...")
        active_actuators = [a for a in ACTUATOR_LIST if a.actuator_id in actuator_ids_to_move]
        
        for actuator in active_actuators:
            if sim_kos:
                await sim_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
            if real_kos:
                await real_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        
        await asyncio.sleep(1)



        # Configure only the motors we need with their gains
        logger.info("Configuring motors for test...")
        logger.info(f"Real robot connection: {real_kos}")
        
        for actuator in active_actuators:
            config_commands = []
            if sim_kos:
                config_commands.append(sim_kos.actuator.configure_actuator(
                    actuator_id=actuator.actuator_id,
                    kp=actuator.kp,
                    kd=actuator.kd,
                    torque_enabled=True,
                ))
            if real_kos:
                config_commands.append(real_kos.actuator.configure_actuator(
                    actuator_id=actuator.actuator_id,
                    kp=actuator.kp,
                    kd=actuator.kd,
                    max_torque=actuator.max_torque,
                    torque_enabled=True,
                ))
            await asyncio.gather(*config_commands)
            logger.info(f"Configured {actuator.name} (ID: {actuator.actuator_id})")

        start_time = time.time()
        next_time = start_time + 1 / 1000  # 1000Hz control rate

        logger.info(f"Starting motion for {duration} seconds...")
        while time.time() - start_time < duration:
            current_time = time.time()
            t = current_time - start_time
            
            # Calculate positions and velocities for each joint based on its profile
            commands = []
            for actuator_id, profile in joint_profiles.items():
                position, velocity = calculate_position_velocity(t, profile)
                commands.append({
                    "actuator_id": actuator_id,
                    "position": position,
                    "velocity": velocity
                })

            # Send commands to simulation and real robot if available
            command_tasks = []
            state_tasks = []
            
            if sim_kos:
                command_tasks.append(sim_kos.actuator.command_actuators(commands))
                state_tasks.append(sim_kos.actuator.get_actuators_state(actuator_ids_to_move))
            
            if real_kos:
                command_tasks.append(real_kos.actuator.command_actuators(commands))
                state_tasks.append(real_kos.actuator.get_actuators_state(actuator_ids_to_move))

            await asyncio.gather(*command_tasks)
            states_list = await asyncio.gather(*state_tasks)

            # Log states separately for sim and real
            state_idx = 0
            if sim_kos:
                for state in states_list[state_idx].states:
                    # Find the corresponding actuator to get kp and kd
                    actuator = next(a for a in ACTUATOR_LIST if a.actuator_id == state.actuator_id)
                    profile = joint_profiles[state.actuator_id]
                    position, velocity = calculate_position_velocity(t, profile)
                    sim_data.log_state(t, state.actuator_id, state.position, state.velocity,
                                    position, velocity, actuator.kp, actuator.kd)
                state_idx += 1
            
            if real_kos:
                for state in states_list[state_idx].states:
                    actuator = next(a for a in ACTUATOR_LIST if a.actuator_id == state.actuator_id)
                    profile = joint_profiles[state.actuator_id]
                    position, velocity = calculate_position_velocity(t, profile)
                    real_data.log_state(t, state.actuator_id, state.position, state.velocity,
                                    position, velocity, actuator.kp, actuator.kd)

            # Sleep to maintain control rate
            if next_time > current_time:
                await asyncio.sleep(next_time - current_time)
            next_time += 1 / 1000

        # Disable motors at the end
        logger.info("Test complete. Disabling motors...")
        await disable_motors(sim_kos, real_kos, actuator_ids_to_move)

    except Exception as e:
        logger.error(f"Error during test: {e}")
        # Make sure motors are disabled even if an error occurs
        await disable_motors(sim_kos, real_kos, actuator_ids_to_move)
        raise  # Re-raise the exception after cleanup

    # Save the logged data to separate JSON files
    output_dir = Path("./test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if sim_data:
        sim_data.save_to_json(str(output_dir / f"sim_actuator_states_{timestamp}.json"))
    if real_data:
        real_data.save_to_json(str(output_dir / f"real_actuator_states_{timestamp}.json"))
    
    logger.info(f"Test data saved to {output_dir}")


async def main() -> None:
    """Runs the main test loop with command line arguments."""
    parser = argparse.ArgumentParser(description="Cable flex life test with multiple joints")
    parser.add_argument("--host", type=str, default="localhost", help="KOS host for simulation")
    parser.add_argument("--port", type=int, default=50051, help="KOS port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--deploy", action="store_true",
                       help="Connect to the real robot (default: simulation only)")
    parser.add_argument("--deploy-only", action="store_true",
                       help="Connect to the real robot only (no simulation)")
    parser.add_argument("--real-host", type=str, default="localhost", 
                       help="KOS host for real robot")
    parser.add_argument("--duration", type=float, default=3600.0,
                       help="Test duration in seconds")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define joint motion profiles for the test
    # This example moves multiple joints with different sine curves
    joint_profiles = {
            31: JointMotionProfile(amplitude=15.0, offset=0.0, frequency=2.0, waveform="sine", time_offset=3.14), 
            32: JointMotionProfile(amplitude=15.0, offset=-20.0, frequency=1.0, waveform="sine", time_offset=3.14),
            33: JointMotionProfile(amplitude=20.0, offset=0.0, frequency=2.0, waveform="sine", time_offset=3.14),
            34: JointMotionProfile(amplitude=30.0, offset=30.0, frequency=2.0, waveform="sine", time_offset=0.0),
            35: JointMotionProfile(amplitude=20.0, offset=0.0, frequency=2.0, waveform="sine", time_offset=0.0),
    }

    try:
        if args.deploy_only and args.deploy:
            logger.warning("Both --deploy and --deploy-only flags were set. Using --deploy-only.")
            args.deploy = False

        if args.deploy_only:
            logger.info(f"Running on real robot only at {args.real_host}")
            async with KOS(ip=args.real_host, port=args.port) as real_kos:
                await test_client(sim_kos=None, real_kos=real_kos, joint_profiles=joint_profiles, duration=args.duration)
        elif not args.deploy:
            logger.info("Running in simulation mode only")
            async with KOS(ip=args.host, port=args.port) as sim_kos:
                await test_client(sim_kos, joint_profiles=joint_profiles, duration=args.duration)
        else:
            logger.info(f"Running in deploy mode with real robot at {args.real_host} and simulation at {args.host}")
            async with KOS(ip=args.host, port=args.port) as sim_kos, \
                    KOS(ip=args.real_host, port=args.port) as real_kos:
                await test_client(sim_kos, real_kos, joint_profiles=joint_profiles, duration=args.duration)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Shutting down safely...")
        # The context manager will handle closing connections
        # The test_client already has exception handling to disable motors
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())