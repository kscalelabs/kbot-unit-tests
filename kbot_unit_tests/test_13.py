from pykos import KOS
import asyncio
import time
import math
kos = KOS("127.0.0.1")

async def move_to_target(current_angle: float, target: float, acceleration: float,
                         V_MAX: float, dt: float, actuator_id: int,
                         profile: str = "linear") -> float:
    """
    Move the actuator from current_angle to target using either a linear (trapezoidal/triangular)
    profile or an S-curve (jerk-limited) profile.
    
    Parameters:
      - acceleration: maximum acceleration (deg/s²)
      - V_MAX: maximum velocity (deg/s)
      - dt: time step (s)
      - profile: "linear" uses the original profile,
                 "scurve" uses a smoothstep-based S-curve profile.
                 
    Returns the final position (which should equal target).
    """
    displacement = target - current_angle
    distance = abs(displacement)
    direction = 1 if displacement >= 0 else -1

    if profile == "linear":
        # --- Linear (trapezoidal) profile (with triangular fallback) ---
        t_accel = V_MAX / acceleration
        d_accel = 0.5 * acceleration * t_accel**2

        if distance >= 2 * d_accel:
            # Trapezoidal case: accelerate to V_MAX, cruise, then decelerate.
            d_flat = distance - 2 * d_accel
            t_flat = d_flat / V_MAX
            total_time = 2 * t_accel + t_flat

            def profile_velocity(t):
                if t < t_accel:
                    return acceleration * t
                elif t < t_accel + t_flat:
                    return V_MAX
                else:
                    return V_MAX - acceleration * (t - t_accel - t_flat)
        else:
            # Triangular case: move too short to reach V_MAX.
            t_phase = math.sqrt(distance / acceleration)
            v_peak = acceleration * t_phase
            total_time = 2 * t_phase

            def profile_velocity(t):
                if t < t_phase:
                    return acceleration * t
                else:
                    return v_peak - acceleration * (t - t_phase)

    elif profile == "scurve":
        # --- S-curve profile using smoothstep for continuous acceleration (and jerk) ---
        T_accel_candidate = 1.5 * V_MAX / acceleration  # duration of acceleration phase
        d_accel_candidate = 0.5 * V_MAX * T_accel_candidate  # distance during acceleration phase

        if distance >= 2 * d_accel_candidate:
            T_accel = T_accel_candidate
            t_flat = (distance - 2 * d_accel_candidate) / V_MAX
            total_time = 2 * T_accel + t_flat

            def profile_velocity(t):
                if t < T_accel:
                    x = t / T_accel
                    return V_MAX * (3 * x**2 - 2 * x**3)
                elif t < T_accel + t_flat:
                    return V_MAX
                elif t < 2 * T_accel + t_flat:
                    x = (2 * T_accel + t_flat - t) / T_accel
                    return V_MAX * (3 * x**2 - 2 * x**3)
                else:
                    return 0.0
        else:
            v_peak = math.sqrt(acceleration * distance / 1.5)
            T_accel = 1.5 * v_peak / acceleration
            total_time = 2 * T_accel

            def profile_velocity(t):
                if t < T_accel:
                    x = t / T_accel
                    return v_peak * (3 * x**2 - 2 * x**3)
                else:
                    x = (2 * T_accel - t) / T_accel
                    return v_peak * (3 * x**2 - 2 * x**3)
    else:
        raise ValueError("Unknown profile type. Choose 'linear' or 'scurve'.")

    steps = int(total_time / dt)
    next_tick = time.perf_counter()
    t = 0.0

    for _ in range(steps):
        velocity = profile_velocity(t)
        signed_velocity = math.copysign(velocity, displacement)
        current_angle += signed_velocity * dt

        await kos.actuator.command_actuators([
            {
                'actuator_id': actuator_id,
                'position': current_angle,
                'velocity': abs(signed_velocity)
            }
        ])

        t += dt
        next_tick += dt
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    current_angle = target
    await kos.actuator.command_actuators([
        {
            'actuator_id': actuator_id,
            'position': current_angle,
            'velocity': 0
        }
    ])
    return current_angle


async def main():
    actuator_ids = [31,32,33,34,35, 41,42,43,44,45] 
    target_positions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #target_positions = [50,0,0,40,0, -50,0,0,-40,0]
    
    V_MAX = 50  # Maximum velocity [deg/s]
    ACCEL = 500  # Maximum acceleration [deg/s^2]
    UPDATE_RATE = 100.0  # Hz
    dt = 1.0 / UPDATE_RATE


    # Configure actuators
    for actuator_id in actuator_ids:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=200.0, kd=5.0,
            max_torque=85, torque_enabled=True
        )

    # Get current positions for each actuator
    current_positions = []
    response = await kos.actuator.get_actuators_state(actuator_ids)
    for state in response.states:
        current_positions.append(state.position)

    print(current_positions)
    # Move each actuator to its target position
    tasks = []
    for actuator_id, current_pos, target_pos in zip(actuator_ids, current_positions, target_positions):
        tasks.append(move_to_target(current_pos, target_pos, ACCEL, V_MAX, dt, actuator_id, profile="scurve"))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
