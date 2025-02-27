import math
import time

#* From Scott's basic motion planning code on Teleop. 
#* Slight change to separate out trajectory generation from execution. 
def find_points_to_target(current_angle: float, target: float, acceleration: float,
                         V_MAX: float, dt: float, actuator_id: int,
                         profile: str = "linear") -> tuple:
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
            print(f"Total time: {total_time}")

            def profile_velocity(t):
                if t < t_accel:
                    return acceleration * t
                elif t < t_accel + t_flat:
                    return V_MAX
                else:
                    return V_MAX - acceleration * (t - t_accel - t_flat)
        else:
            print("Triangular case!")
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
    
    # Create lists to store trajectory data
    trajectory_angles = [current_angle]
    trajectory_velocities = []
    trajectory_times = [0.0]
    
    t = 0.0

    for _ in range(steps):
        velocity = profile_velocity(t)
        signed_velocity = math.copysign(velocity, displacement)
        current_angle += signed_velocity * dt
        
        # Store trajectory data
        trajectory_velocities.append(signed_velocity)
        trajectory_angles.append(current_angle)
        t += dt
        trajectory_times.append(t)
        next_tick += dt

    # Set final angle to exactly target
    trajectory_angles[-1] = target

    return trajectory_angles, trajectory_velocities, trajectory_times
