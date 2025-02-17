"""Interactive example script for a command to keep the robot balanced."""

import argparse
import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass

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
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
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

    Actuator(31, 0, 250.0, 30.0, 120.0, "left_hip_pitch_04"),
    Actuator(32, 4, 150.0, 8.0, 60.0, "left_hip_roll_03"),
    Actuator(33, 8, 150.0, 8.0, 60.0, "left_hip_yaw_03"),
    Actuator(34, 12, 200.0, 8.0, 120.0, "left_knee_04"),
    Actuator(35, 16, 80.0, 10.0, 17.0, "left_ankle_02"),

    Actuator(41, 2, 250.0, 30.0, 120.0, "right_hip_pitch_04"),
    Actuator(42, 6, 150.0, 8.0, 60.0, "right_hip_roll_03"),
    Actuator(43, 10, 150.0, 8.0, 60.0, "right_hip_yaw_03"),
    Actuator(44, 14, 200.0, 8.0, 120.0, "right_knee_04"),
    Actuator(45, 18, 80.0, 10.0, 17.0, "right_ankle_02"),  
]


async def configure_robot(kos: KOS, is_real: bool) -> None:
    for actuator in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id = actuator.actuator_id,
            kp = actuator.kp,
            kd = actuator.kd,
            max_torque = actuator.max_torque,
        )
    pass


async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    # await test_client(host=args.host, port=args.port)
    sim_process = subprocess.Popen(["kos-sim", "kbot-v1", "--no-gravity"])
    time.sleep(2)
    try:
        print("Running on both simulator and real robots simultaneously...")
        async with KOS(ip="localhost", port=50051) as sim_kos, KOS(ip="10.33.11.170", port=50051) as real_kos:
            await sim_kos.sim.reset()

            sim_kos = await configure_robot(sim_kos, False)
            real_kos = await configure_robot(real_kos, True)

        print("Homing...")
        homing_command = [{"actuator_id": actuator.actuator_id, "position": 0.0} for actuator in ACTUATOR_LIST]
        await asyncio.gather(
            sim_kos.actuator.command_actuators(homing_command),
            real_kos.actuator.command_actuators(homing_command),
        )

        await asyncio.sleep(2)

        for actuator in ACTUATOR_LIST:
            print(f"Teesting {actuator.actuator_id}...")
            test_angle = -45.0

            real_command = {
                "actuator_id": actuator.actuator_id,
                "position": test_angle,
            }

            sim_command = {
                "actuator_id": actuator.actuator_id,
                "position": test_angle,
            }

            await asyncio.gather(
                sim_kos.actuator.command_actuator(sim_command),
                real_kos.actuator.command_actuator(real_command),
            )

            await asyncio.sleep(2)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sim_process.terminate()
        sim_process.wait()



if __name__ == "__main__":
    asyncio.run(main())