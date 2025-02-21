"""Script for making the robot stand up from lying face-down on the ground."""

import argparse
import asyncio
import logging
from dataclasses import dataclass

import colorlogging
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


ACTUATOR_LIST: list[Actuator] = [
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
    Actuator(31, 0, 250.0, 30.0, 120.0),  # left_hip_pitch_04
    Actuator(32, 4, 150.0, 8.0, 60.0),  # left_hip_roll_03
    Actuator(33, 8, 250.0, 8.0, 60.0),  # left_hip_yaw_03
    Actuator(34, 12, 200.0, 8.0, 120.0),  # left_knee_04
    Actuator(35, 16, 80.0, 10.0, 17.0),  # left_ankle_02
    Actuator(41, 2, 250.0, 30.0, 120.0),  # right_hip_pitch_04
    Actuator(42, 6, 150.0, 8.0, 60.0),  # right_hip_roll_03
    Actuator(43, 10, 250.0, 8.0, 60.0),  # right_hip_yaw_03
    Actuator(44, 14, 200.0, 8.0, 120.0),  # right_knee_04
    Actuator(45, 18, 80.0, 10.0, 17.0),  # right_ankle_02
]

# Add this mapping at the top of the file with other constants
ACTUATOR_TO_JOINT_MAP = {
    11: "right_shoulder_pitch_03",
    12: "right_shoulder_roll_03", 
    13: "right_shoulder_yaw_02",
    14: "right_elbow_02",
    15: "right_wrist_02",
    21: "left_shoulder_pitch_03",
    22: "left_shoulder_roll_03",
    23: "left_shoulder_yaw_02",
    24: "left_elbow_02",
    25: "left_wrist_02",
    31: "right_hip_pitch_04",
    32: "right_hip_roll_03",
    33: "right_hip_yaw_03",
    34: "right_knee_04",
    35: "right_ankle_02",
    41: "left_hip_pitch_04",
    42: "left_hip_roll_03",
    43: "left_hip_yaw_03",
    44: "left_knee_04",
    45: "left_ankle_02",
}


async def stand_up(kos: KOS) -> None:
    """Execute stand-up sequence from prone position."""
    # Phase 0: Very gradual initial leg folding
    logger.info("Phase 0: Initial leg folding")

    await kos.actuator.command_actuators(
        [
            {"actuator_id": 32, "position": -70.0},   # right_hip_roll_03
            {"actuator_id": 42, "position": 70.0},    # left_hip_roll_03
        ]
    )
    await asyncio.sleep(1.0)
    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
            {"actuator_id": 12, "position": -90.0},    # right_shoulder_roll_03
            {"actuator_id": 14, "position": 0.0},       # right_elbow_02
            {"actuator_id": 22, "position": 90.0},     # left_shoulder_roll_03
            {"actuator_id": 24, "position": 0.0},       # left_elbow_02
            {"actuator_id": 34, "position": 180.0},     # right_knee_04
            {"actuator_id": 44, "position": -180.0},    # left_knee_04
        ]
    )
    await asyncio.sleep(1.0)
    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
            # i want elbow to be 0
            {"actuator_id": 14, "position": -90.0},     # right_elbow_02
            {"actuator_id": 24, "position": 90.0},     # left_elbow_02
            {"actuator_id": 31, "position": 90.0},     # right_hip_pitch_04
            {"actuator_id": 41, "position": -90.00},   # left_hip_pitch_04
            {"actuator_id": 35, "position": -25.0},    # right_ankle_02
            {"actuator_id": 45, "position": 25.0},     # left_ankle_02
        ]
    )
    
   
    
    await asyncio.sleep(7.0)    
    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
  # i want the hip pitch to be 0
            {"actuator_id": 31, "position": 120.0},     # right_hip_pitch_04
            {"actuator_id": 41, "position": -120.0},   # left_hip_pitch_04
            # i want shoulder roll to be 0
            {"actuator_id": 12, "position": -55.0},     # right_shoulder_roll_03
            {"actuator_id": 22, "position": 55.0},   # left_shoulder_roll_03
        ]
    )
    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
            # i want hip yaw to be 0
            {"actuator_id": 33, "position": -90.0},     # right_hip_yaw_03
            {"actuator_id": 43, "position": 90.0},   # left_hip_yaw_03
            # {"actuator_id": 35, "position": -25.0},    # right_ankle_02
            # {"actuator_id": 45, "position": 25.0},     # left_ankle_02
        ]
    )
    await asyncio.sleep(2.0)
    logger.info("Phase 4: Standing up")


    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [

            # i want the hip pitch to be 0
            {"actuator_id": 31, "position": 120.0},     # right_hip_pitch_04
            {"actuator_id": 41, "position": -120.0},   # left_hip_pitch_04
            # i want shoulder roll to be 0
            {"actuator_id": 12, "position": -45.0},     # right_shoulder_roll_03
            {"actuator_id": 22, "position": 45.0},   # left_shoulder_roll_03
        ]
    )
    await asyncio.sleep(2.0)
    logger.info("Phase 4: Standing up")





    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [

            # i want the elbows to be 0
            {"actuator_id": 12, "position": -25.0},     # right_shoulder_roll_03
            {"actuator_id": 22, "position": 25.0},   # left_shoulder_roll_03
            # i want the elbows to be 0
            {"actuator_id": 14, "position": -65.0},     # right_elbow_02
            {"actuator_id": 24, "position": 65.0},   # left_elbow_02
            # i want hip roll to be 0
            # {"actuator_id": 33, "position": -10.0},   # left hip roll
            # {"actuator_id": 43, "position": 10.0},    # right hip roll
        ]
    )
    await asyncio.sleep(7.0)
    logger.info("Phase 4: Standing up")




    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [

            # i want the elbows to be 0
            {"actuator_id": 14, "position": -120.0},     # right_elbow_02
            {"actuator_id": 24, "position": 120.0},   # left_elbow_02
        ]
    )
    await asyncio.sleep(2.0)
    logger.info("Phase 4: Standing up")

    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
            {"actuator_id": 32, "position": -30.0},   # left hip roll
            {"actuator_id": 42, "position": 30.0},  # right hip roll

            {"actuator_id": 34, "position": 180.0},     # right_knee_04
            {"actuator_id": 44, "position": -180.0},    # left_knee_04
            # i want hi pitch to be 0
            {"actuator_id": 31, "position": 140.0},     # left hip pitch
            {"actuator_id": 41, "position": -140.0},    # right hip pitch
            # i want the ankle to be 0
            {"actuator_id": 35, "position": -55.0},     # left ankle
            {"actuator_id": 45, "position": 55.0},    # right ankle
            # i want the shoulder pitch to be 0
            {"actuator_id": 11, "position": 80.0},     # left shoulder pitch
            {"actuator_id": 21, "position": -80.0},    # right shoulder pitch
            # i want shoulder roll to be 0
            {"actuator_id": 12, "position": 0.0},     # left shoulder roll
            {"actuator_id": 22, "position": 0.0},    # right shoulder roll

        
        ]
    )



    await asyncio.sleep(1.0)
    logger.info("Phase 4: Standing up")


    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
             {"actuator_id": 14, "position": 0.0},     # left elbow
             {"actuator_id": 24, "position": 0.0},     # right elbow   
        ]
    )
    await asyncio.sleep(2.0)
    logger.info("Phase 4: Standing up")

    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
            {"actuator_id": 33, "position": 5.0},   # left hip roll
            {"actuator_id": 43, "position": -5.0},    # right hip roll
        ]
    )
    await asyncio.sleep(2.0)



async def test_client(host: str = "localhost", port: int = 50051) -> None:
    logger.info("Starting stand-up client...")

    async with KOS(ip=host, port=port) as kos:
        # # Start the robot lying down with zeroed joints, but slightly higher
        # init_xyz = [0.0, 0.0, 0.4]  # Increased height to reduce ground friction
        # init_quat = list(R.from_euler("xyz", [00.0, 0.0, 0.0], degrees=True).as_quat())
        # init_joints = [0.0] * len(ACTUATOR_LIST)
        # # breakpoint()
        # Reset the simulation.
        # await kos.sim.resegit(
        #     initial_state={
        #         "qpos": init_xyz + init_quat + init_joints,
        #         "qvel": [0.0] * len(ACTUATOR_LIST),
        #     }
        # )
        # await kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})


        # Configure all actuators with higher gains for legs
        for actuator in ACTUATOR_LIST:
            kp = actuator.kp
            kd = actuator.kd
            # Increase gains for leg joints
            if actuator.actuator_id in [31, 34, 41, 44]:  # hip and knee joints
                kp = 250.0  # Increased from 250.0
                kd = 30.0   # Increased from 30.0
            
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=kp,
                kd=kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )

        await kos.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 0.4},
            quat={"x": -0.707, "y": 0.0, "z": 0.0, "w": 0.707},
            joints=[
                {
                    "name": ACTUATOR_TO_JOINT_MAP[actuator.actuator_id],
                    "pos": 0.0,
                }
                for actuator in ACTUATOR_LIST
            ],
        )
        await asyncio.sleep(0.2)

        # Remove the infinite loop
        commands = []
        for actuator in ACTUATOR_LIST:
            commands.append({"actuator_id": actuator.actuator_id, "position": 0})
        await kos.actuator.command_actuators(commands)
        await asyncio.sleep(0.2)

        # Execute stand-up sequence
        await stand_up(kos)
        logger.info("Stand-up sequence completed")


async def main() -> None:
    """Run the stand-up sequence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    await test_client(host=args.host, port=args.port)


if __name__ == "__main__":
    # python -m examples.kbot.stand_up
    asyncio.run(main())