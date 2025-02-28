import argparse
import asyncio
import logging
import time
from pathlib import Path

import colorlogging
from pykos import KOS

from kbot_cycle_tests.trajectory_execution_primitive import (
    move_actuators_with_trajectory,
    setup_actuator,
    disable_actuator,
    ACTUATOR_LIST
)

logger = logging.getLogger(__name__)

async def test_actuator_pairs(sim_kos: KOS, real_kos: KOS = None) -> None:
    """
    Test moving different actuator pairs sequentially with breakpoints between each pair.
    Each pair of actuators will move to 15 degrees, pause for user inspection, then be reset to 0.
    """
    # Reset the simulation to a standing position
    await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
    
    # Define actuator pairs to test
    actuator_pairs = [
        # Knees
        {"name": "Knees", "ids": [34, 44], "description": "Left and Right Knee"},
        # Ankles
        {"name": "Ankles", "ids": [35, 45], "description": "Left and Right Ankle"},
        # Hip Pitch
        {"name": "Hip Pitch", "ids": [31, 41], "description": "Left and Right Hip Pitch"},
        # Hip Roll
        {"name": "Hip Roll", "ids": [32, 42], "description": "Left and Right Hip Roll"},
        # Hip Yaw
        {"name": "Hip Yaw", "ids": [33, 43], "description": "Left and Right Hip Yaw"},
        # Shoulder Pitch
        {"name": "Shoulder Pitch", "ids": [11, 21], "description": "Left and Right Shoulder Pitch"},
        # Shoulder Roll
        {"name": "Shoulder Roll", "ids": [12, 22], "description": "Left and Right Shoulder Roll"},
        # Shoulder Yaw
        {"name": "Shoulder Yaw", "ids": [13, 23], "description": "Left and Right Shoulder Yaw"},
        # Elbows
        {"name": "Elbows", "ids": [14, 24], "description": "Left and Right Elbow"},
        # Wrists
        {"name": "Wrists", "ids": [15, 25], "description": "Left and Right Wrist"},
    ]
    
    # Test each actuator pair
    for pair in actuator_pairs:
        logger.info(f"Testing {pair['name']} actuators: {pair['description']}")
        
        # Setup the actuators
        for actuator_id in pair["ids"]:
            # Find the actuator in ACTUATOR_LIST
            actuator = next((a for a in ACTUATOR_LIST if a.actuator_id == actuator_id), None)
            if actuator:
                await setup_actuator(
                    sim_kos=sim_kos,
                    actuator_id=actuator_id,
                    kp=actuator.kp,
                    kd=actuator.kd,
                    max_torque=actuator.max_torque,
                    real_kos=real_kos
                )
            else:
                logger.error(f"Actuator ID {actuator_id} not found in ACTUATOR_LIST")
                continue
        
        # Move actuators to 15 degrees
        logger.info(f"Moving {pair['name']} to 15 degrees")
        await move_actuators_with_trajectory(
            sim_kos=sim_kos,
            actuator_ids=pair["ids"],
            target_angles=[-15.0, -15.0],
            real_kos=real_kos
        )
        
        # Breakpoint - wait for user input
        logger.info(f"BREAKPOINT: {pair['name']} at 15 degrees. Press Enter to continue...")
        input()
        
        # Move actuators back to 0 degrees
        logger.info(f"Moving {pair['name']} back to 0 degrees")
        await move_actuators_with_trajectory(
            sim_kos=sim_kos,
            actuator_ids=pair["ids"],
            target_angles=[0.0, 0.0],
            real_kos=real_kos
        )
        
        # Disable the actuators
        for actuator_id in pair["ids"]:
            await disable_actuator(sim_kos, actuator_id, real_kos)
        
        # Wait a moment before moving to the next pair
        await asyncio.sleep(1)
    
    logger.info("Test completed successfully!")

async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser(description="Test actuator pairs with breakpoints")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--deploy", action="store_true",
                       help="Connect to the real robot (default: simulation only)")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    if not args.deploy:
        # Simulation only
        async with KOS(ip=args.host, port=args.port) as sim_kos:
            await test_actuator_pairs(sim_kos)
    else:
        # Both simulation and real robot
        async with KOS(ip=args.host, port=args.port) as sim_kos, \
                   KOS(ip="100.117.248.15", port=args.port) as real_kos:
            await test_actuator_pairs(sim_kos, real_kos)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
