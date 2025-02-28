import argparse
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass

import colorlogging
logger = logging.getLogger(__name__)
from pykos import KOS

from kbot_cycle_tests.motion_planning_primitive import find_points_to_target
from kbot_cycle_tests.trajectory_execution_primitive import (
    move_actuators_with_trajectory,
    setup_actuator,
    disable_actuator,
    ACTUATOR_LIST
)
from kbot_cycle_tests.plot_motion_plan import plot_motion_plan


async def setup_actuators(active_actuators: list[int], sim_kos: KOS, real_kos: KOS = None) -> None:
    """
    Setup the actuators.
    """
    for actuator in ACTUATOR_LIST:
        if actuator.actuator_id in active_actuators:
            await setup_actuator(sim_kos, actuator.actuator_id, actuator.kp, actuator.kd, actuator.max_torque, real_kos)

async def squat(sim_kos: KOS, real_kos: KOS = None) -> None:
    """
    Start the robot in a squatting position and then stand it up.
    This approach helps us focus on getting the correct actuator positions first.
    """
    # Reset the simulation
    await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
    
    # Define the actuator IDs we'll be using for the squat
    # Left leg: hip pitch (31), knee (34), ankle (35)
    # Right leg: hip pitch (41), knee (44), ankle (45)
    actuator_ids = [31, 41, 34, 44, 35, 45]
    
    # Setup all the actuators we'll be using
    logger.info("Setting up actuators...")
    await setup_actuators(actuator_ids, sim_kos, real_kos)
    
    # Initial squat position
    logger.info("Moving to deep squat position...")
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=actuator_ids,  # [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        target_angles=[-55.0, -55.0, -120.0, -120.0, -40.0, -40.0],  # Deep squat position
        real_kos=real_kos
    )
    
    await asyncio.sleep(1)
    
    # WAYPOINT 1: Begin standing up - knees start first
    logger.info("Beginning to stand up - knees first...")
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=actuator_ids,
        target_angles=[-55.0, -55.0, -90.0, -90.0, -30.0, -30.0],  # Knees start straightening while hips stay
        real_kos=real_kos
    )
    
    await asyncio.sleep(0.5)
    
    # WAYPOINT 2: Continue standing - knees and hips move together
    logger.info("Continuing to stand - coordinated movement...")
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=actuator_ids,
        target_angles=[-40.0, -40.0, -60.0, -60.0, -20.0, -20.0],  # Both knees and hips moving
        real_kos=real_kos
    )
    
    await asyncio.sleep(0.5)
    
    # WAYPOINT 3: Mid-standing position
    logger.info("Mid-standing position...")
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=actuator_ids,
        target_angles=[-25.0, -25.0, -30.0, -30.0, -10.0, -10.0],  # Continue coordinated movement
        real_kos=real_kos
    )
    
    await asyncio.sleep(0.5)
    
    # WAYPOINT 4: Almost standing
    logger.info("Almost standing...")
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=actuator_ids,
        target_angles=[-10.0, -10.0, -15.0, -15.0, -5.0, -5.0],  # Almost straight
        real_kos=real_kos
    )
    
    await asyncio.sleep(0.5)
    
    # WAYPOINT 5: Fully standing
    logger.info("Standing up completely...")
    await move_actuators_with_trajectory(
        sim_kos=sim_kos,
        actuator_ids=actuator_ids,
        target_angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Standing position
        real_kos=real_kos
    )
    
    # Wait to observe the standing position
    await asyncio.sleep(1)
    
    # Disable all the actuators
    for actuator_id in actuator_ids:
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
            await squat(sim_kos)
    else:
        async with KOS(ip=args.host, port=args.port) as sim_kos, \
                   KOS(ip="100.117.248.15", port=args.port) as real_kos:
            await squat(sim_kos, real_kos)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())


