"""Logging script for testing armature."""
#! MAKE SURE BOTH ARE OFF IN THE SAME WAY IN KOS SIM AND REAL
""" REQUIRED ADJUSTMENTS:
Alter the MJCF to contain the following after </worldbody>:
    <equality>
            <weld body1="body1-part" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="leg0_shell" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="leg0_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="leg1_shell" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="leg1_shell3" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="leg2_shell" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="leg2_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="shoulder" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="shoulder_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="arm1_top" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="arm1_top_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="arm2_shell" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="arm2_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="arm3_shell" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/> 
            <weld body1="hand_shell" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="hand_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="hand_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="hand_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="hand_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="hand_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            <weld body1="hand_shell_2" body2="world" solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </equality>

    ALTER the mjcf so that leg3 and foot3 to have very small mass and diag inertia (won't let you enter 0, get as close as possible)

    ALTER the mjcf so that leg3 and foot3 and adjacent (leg2, leg1) HAVE NO COLLISION

    ALTER the mjcf so that the motor being used has no joint limit

    CHECK the mjcf to see what the 'actual' friction and armature values are.

ALTER the kos-sim definition to have mujoco-scene be empty (rather than "smooth") 

MAKE SURE that you run it --no-gravity
"""



import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
import json
from typing import Dict, List

import colorlogging
import numpy as np
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float


@dataclass
class StateLog:
    time: float
    actuator_id: int
    position: float
    velocity: float
    torque: float
    commanded_position: float
    commanded_velocity: float
    commanded_torque: float
    kp: float
    kd: float


class TestData:
    def __init__(self):
        self.states: List[StateLog] = []
    
    def log_state(self, time: float, actuator_id: int, position: float, velocity: float, torque: float, 
                 commanded_position: float, commanded_velocity: float, commanded_torque: float, kp: float, kd: float):
        self.states.append(StateLog(time, actuator_id, position, velocity, torque,
                                  commanded_position, commanded_velocity, commanded_torque, kp, kd))
    
    def save_to_json(self, filename: str):
        data = [{"time": s.time, "actuator_id": s.actuator_id,
                 "kp": s.kp,
                 "kd": s.kd,
                 "position": s.position, "velocity": s.velocity,
                 "torque": s.torque,
                 "commanded_position": s.commanded_position,
                 "commanded_velocity": s.commanded_velocity,
                 "commanded_torque": s.commanded_torque}
                for s in self.states]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


ACTUATOR_LIST: list[Actuator] = [
    # actuator id, nn id, kp, kd, max_torque
    Actuator(11, 1, 150.0, 8.0, 60.0),  # left_shoulder_pitch_03
    Actuator(12, 5, 150.0, 8.0, 60.0),  # left_shoulder_roll_03
    Actuator(13, 9, 50.0, 5.0, 17.0),  # left_shoulder_yaw_02
    Actuator(14, 13, 50.0, 5.0, 17.0),  # left_elbow_02
    Actuator(15, 17, 20.0, 2.0, 17.0),  # left_wrist_02
    Actuator(21, 3, 150.0, 8.0, 60.0),  # right_shoulder_pitch_03
    Actuator(22, 7, 150.0, 8.0, 60.0),  # right_shoulder_roll_03
    Actuator(23, 11, 50.0, 5.0, 17.0),  # right_shoulder_yaw_02
    Actuator(24, 15, 50.0, 5.0, 17.0),  # right_elbow_02
    Actuator(25, 19, 20.0, 2.0, 17.0),  # right_wrist_02
    Actuator(31, 0, 85.0, 3.0, 80.0),  # left_hip_pitch_04 (RS04_Pitch)
    Actuator(32, 4, 85.0, 2.0, 60.0),  # left_hip_roll_03 (RS03_Roll)
    Actuator(33, 8, 30.0, 2.0, 60.0),  # left_hip_yaw_03 (RS03_Yaw)
    Actuator(34, 12, 60.0, 2.0, 80.0),  # left_knee_04 (RS04_Knee)
    Actuator(35, 16, 80.0, 1.0, 17.0),  # left_ankle_02 (RS02)
    Actuator(41, 2, 85.0, 3.0, 80.0),  # right_hip_pitch_04 (RS04_Pitch)
    Actuator(42, 6, 85.0, 2.0, 60.0),  # right_hip_roll_03 (RS03_Roll)
    Actuator(43, 10, 30.0, 2.0, 60.0),  # right_hip_yaw_03 (RS03_Yaw)
    
    #Actuator(44, 14, 100.0, 7.0, 80.0),  # right_knee_04 (RS04_Knee)
    Actuator(44, 14, 0, 0, 80.0),  # right_knee_04 (RS04_Knee)


    Actuator(45, 18, 80.0, 1.0, 17.0),  # right_ankle_02 (RS02)
]

# Define the actuators we want to move
ACTUATORS_TO_MOVE = [44]  # right knee RS04

async def test_client(sim_kos: KOS, real_kos: KOS = None) -> None:
    logger.info("Starting test client...")
    
    # Initialize separate test data loggers for sim and real
    sim_data = TestData()
    real_data = TestData() if real_kos else None

    # Test parameters
    torque_direction = 1 
    step_current = 0.03 # Amps

    step_time_duration = .5    # seconds
    step_time_delay = 1.0   # seconds
    step_time_shutoff = 14 # seconds
    duration = step_time_delay + step_time_duration + step_time_shutoff   # seconds
    Km_RS04 = 2.1 # Nm/Ampere_rms https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS04/RS04_instruction_manual_En_241217.pdf

    
    # Reset the simulation
    await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0] + [0.0] * 20})


    # First disable all motors
    logger.info("Disabling motors...")
    for actuator in ACTUATOR_LIST:
        await sim_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        if real_kos:
            await real_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
    
    await asyncio.sleep(1)

    # Configure motors with their gains
    logger.info("Configuring motors...")
    for actuator in ACTUATOR_LIST:
        config_commands = []
        config_commands.append(sim_kos.actuator.configure_actuator(
            actuator_id=actuator.actuator_id,
            torque_enabled=True,
            kp=0.0,#actuator.kp,
            kd=0.0,#actuator.kd,
            max_torque=actuator.max_torque,
        ))
        if real_kos:
            config_commands.append(real_kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,          # P gain
                kd=actuator.kd,          # D gain
                max_torque=actuator.max_torque,  # torque limit
                torque_enabled=True,
                control_mode="current"
            ))
        await asyncio.gather(*config_commands)

    start_time = time.time()
    next_time = start_time + 1 / 100  # 100Hz control rate

    while time.time() - start_time < duration:
        current_time = time.time()
        t = current_time - start_time
        
        # Define step function input torque
        #   Allow for steady state to arise 
        torque = 0.0
        if t > step_time_delay + step_time_duration:
            torque = 0.0
        elif t > step_time_delay:
            torque = step_current * Km_RS04 * torque_direction

        # Calculate sine wave position with offset
        #angular_freq = 2 * np.pi * frequency
        #position = offset + amplitude * np.sin(angular_freq * t)
        # velocity = amplitude * angular_freq * np.cos(angular_freq * t)
        #velocity = 0.0

        position = 0.0
        velocity = 0.0

        # Filter actuator list to only include actuators we want to move
        active_actuators = [actuator for actuator in ACTUATOR_LIST if actuator.actuator_id in ACTUATORS_TO_MOVE]

        commands = [
            {
                "actuator_id": actuator.actuator_id, 
                #"position": position,     # Target position in degrees
                #"velocity": velocity      # Target velocity in deg/s
                "torque": torque
            }
            for actuator in active_actuators
        ]

        command_tasks = [sim_kos.actuator.command_actuators(commands)]
        state_tasks = [sim_kos.actuator.get_actuators_state(actuator.actuator_id for actuator in active_actuators)]
        
        if real_kos:
            command_tasks.append(real_kos.actuator.command_actuators(commands))
            state_tasks.append(real_kos.actuator.get_actuators_state(actuator.actuator_id for actuator in active_actuators))

        await asyncio.gather(*command_tasks)
        states_list = await asyncio.gather(*state_tasks)

        # Log states separately for sim and real
        for state in states_list[0].states:
            # Find the actuator config for this state
            actuator_config = next(a for a in ACTUATOR_LIST if a.actuator_id == state.actuator_id)
            sim_data.log_state(t, state.actuator_id, state.position, state.velocity, state.torque,
                             position, velocity, torque, actuator_config.kp, actuator_config.kd)
        
        if real_kos and len(states_list) > 1:
            for state in states_list[1].states:
                # Find the actuator config for this state
                actuator_config = next(a for a in ACTUATOR_LIST if a.actuator_id == state.actuator_id)
                real_data.log_state(t, state.actuator_id, state.position, state.velocity, state.torque,
                                  position, velocity, torque, actuator_config.kp, actuator_config.kd)

        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)
        next_time += 1 / 100

        # Right after sending commands:
        logger.debug(f"Sent commands: {commands}")
        # After receiving states:
        logger.debug(f"Received states: {states_list[0].states}")

    # Disable motors at the end
    logger.info("Disabling motors...")
    for actuator in ACTUATOR_LIST:
        disable_commands = [
            sim_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        ]
        if real_kos:
            disable_commands.append(
                real_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
            )
        await asyncio.gather(*disable_commands)

    # Save the logged data to separate JSON files
    sim_data.save_to_json("./utils/test07_sim_actuator_states.json")
    if real_data:
        real_data.save_to_json("./utils/test07_real_actuator_states.json")

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
            await test_client(sim_kos)
    else:
        async with KOS(ip=args.host, port=args.port) as sim_kos, \
                   KOS(ip="100.89.14.31", port=args.port) as real_kos:
            await test_client(sim_kos, real_kos)


if __name__ == "__main__":
    # python -m examples.kbot.balancing
    asyncio.run(main())