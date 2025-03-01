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
    setup_actuators,
    disable_actuators,
    ACTUATOR_LIST
)
from kbot_cycle_tests.plot_motion_plan import plot_motion_plan


async def go_to_zero(sim_kos: KOS = None, real_kos: KOS = None) -> None:
    """
    Move all robot actuators to the zero position.
    
    Args:
        sim_kos: KOS instance for simulation. Can be None if running only on real robot.
        real_kos: KOS instance for real robot. Can be None if running only in simulation.
    """
    # Reset the simulation if we're using it
    if sim_kos:
        await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
    
    # Define all actuator IDs we want to move to zero
    # Arms: shoulders, elbows, wrists
    # Legs: hips, knees, ankles
    actuator_ids = [
        # Left arm
        # 11, 12, 13, 14, 15,
        # Right arm
        21, 22, 23, 24, 25,
        # Left leg
        31, 32, 33, 34, 35,
        # Right leg
        41, 42, 43, 44, 45
    ]
    
    # Get the actuator objects for the IDs we'll be using
    zero_actuators = [ACTUATOR_LIST[actuator_id] for actuator_id in actuator_ids]
    
    # Setup all the actuators we'll be using at once
    logger.info("Setting up actuators...")
    await setup_actuators(sim_kos, zero_actuators, real_kos)
    
    try:
        # Get the current positions of all actuators
        kos_to_use = sim_kos if sim_kos is not None else real_kos
        if kos_to_use is None:
            raise ValueError("Both sim_kos and real_kos cannot be None")
            
        state_response = await kos_to_use.actuator.get_actuators_state(actuator_ids)
        current_angles = [state.position for state in state_response.states]
        
        logger.info(f"Current actuator angles: {current_angles}")
        
        # Create a list of zero angles for all actuators
        target_angles = [0.0] * len(actuator_ids)
        
        # Move all actuators to zero position
        logger.info("Moving all actuators to zero position...")
        await move_actuators_with_trajectory(
            sim_kos=sim_kos,
            actuator_ids=actuator_ids,
            target_angles=target_angles,
            real_kos=real_kos
        )
        
        # Wait to observe the zero position
        await asyncio.sleep(1)
        
    except asyncio.CancelledError:
        logger.warning("Operation was cancelled - safely disabling actuators before exit")
    except Exception as e:
        logger.error(f"Error during go_to_zero operation: {e}")
    finally:
        # Disable all the actuators at once - this will always run, even if there's an exception
        logger.info("Disabling actuators...")
        await disable_actuators(sim_kos, actuator_ids, real_kos)


async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--real-host", type=str, default="100.117.248.15",
                       help="IP address of the real robot")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--deploy", action="store_true",
                       help="Connect to the real robot alongside simulation")
    parser.add_argument("--deploy-only", action="store_true",
                       help="Connect only to the real robot without simulation")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    try:
        # Check for conflicting flags
        if args.deploy_only and args.deploy:
            logger.warning("Both --deploy and --deploy-only flags were set. Using --deploy-only.")
            args.deploy = False
            
        if args.deploy_only:
            logger.info(f"Running on real robot only at {args.real_host}")
            async with KOS(ip=args.real_host, port=args.port) as real_kos:
                await go_to_zero(sim_kos=None, real_kos=real_kos)
        elif not args.deploy:
            logger.info("Running in simulation mode only")
            async with KOS(ip=args.host, port=args.port) as sim_kos:
                await go_to_zero(sim_kos=sim_kos, real_kos=None)
        else:
            logger.info(f"Running in deploy mode with real robot at {args.real_host} and simulation at {args.host}")
            async with KOS(ip=args.host, port=args.port) as sim_kos, \
                       KOS(ip=args.real_host, port=args.port) as real_kos:
                await go_to_zero(sim_kos=sim_kos, real_kos=real_kos)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received - shutting down")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
