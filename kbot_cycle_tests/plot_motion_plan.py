import numpy as np
import matplotlib.pyplot as plt
import math
from motion_planning_primitive import find_points_to_target

# Parameters for the trajectory
current_angle = 0.0
target_angle = 5
acceleration = 100.0  # rad/s²
V_MAX = 10.0  # rad/s
dt = 0.01  # seconds
actuator_id = 1  # arbitrary ID

# Generate trajectories for both profile types
profiles = ["linear", "scurve"]
trajectory_data = {}

for profile in profiles:
    # Generate trajectory
    angles, velocities, times = find_points_to_target(
        current_angle=current_angle,
        target=target_angle,
        acceleration=acceleration,
        V_MAX=V_MAX,
        dt=dt,
        actuator_id=actuator_id,
        profile=profile
    )
    
    trajectory_data[profile] = {
        "angles": angles,
        "velocities": velocities,
        "times": times
    }

# Create plots
plt.figure(figsize=(12, 10))

# Plot position trajectories
plt.subplot(2, 1, 1)
for profile, data in trajectory_data.items():
    plt.plot(data["times"], data["angles"], label=f"{profile.capitalize()} Profile")
plt.axhline(y=target_angle, color='r', linestyle='--', label='Target Angle')
plt.title('Position Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.legend()

# Plot velocity trajectories
plt.subplot(2, 1, 2)
for profile, data in trajectory_data.items():
    plt.plot(data["times"][:-1], data["velocities"], label=f"{profile.capitalize()} Profile")
plt.title('Velocity Profile')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (rad/s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('trajectory_plot.png')
plt.show()

# Print some statistics
for profile, data in trajectory_data.items():
    print(f"\n{profile.capitalize()} Profile Statistics:")
    print(f"  Total time: {data['times'][-1]:.3f} seconds")
    print(f"  Max velocity: {max(abs(v) for v in data['velocities']):.3f} rad/s")
    print(f"  Final position: {data['angles'][-1]:.3f} rad ({data['angles'][-1]/math.pi:.3f}π)")
