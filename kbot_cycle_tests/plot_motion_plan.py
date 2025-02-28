import numpy as np
import matplotlib.pyplot as plt
import time
from motion_planning_primitive import find_points_to_target

def plot_motion_plan(current_angle: float = None, target_angle: float = None, profile: str = "scurve", save_file: bool = False, show_plot: bool = True,
                     actuator_id: int = None, acceleration: float = 100.0, v_max: float = 30.0, update_rate: float = 100.0):
    """
    Generate and plot motion trajectories for a motion profile.
    
    Parameters:
    -----------
    current_angle : float
        Starting angle in degrees
    target_angle : float
        Target angle in degrees
    acceleration : float
        Maximum acceleration in deg/s²
    v_max : float
        Maximum velocity in deg/s
    update_rate : float
        Update rate in Hz
    profile : str
        Profile type to plot. Default is "scurve"
    save_file : bool
        Whether to save the plot. Default is False.
    show_plot : bool
        Whether to display the plot. Default is True.
    actuator_id : int
        ID of the actuator to plot. Default is None.
    Returns:
    --------
    dict
        Dictionary containing trajectory data for the profile
    """
    # Initialize trajectory data dictionary
    trajectory_data = {}

    # Generate trajectory
    angles, velocities, times = find_points_to_target(
        current_angle=current_angle,
        target=target_angle,
        acceleration=acceleration,
        V_MAX=v_max,
        update_rate=update_rate,
        profile=profile
    )
    
    trajectory_data[profile] = {
        "angles": angles,
        "velocities": velocities,
        "times": times
    }

    # Create plots
    plt.figure(figsize=(12, 10))

    # Plot position trajectory
    plt.subplot(2, 1, 1)
    plt.plot(times, angles, label=f"{profile.capitalize()} Profile")
    plt.axhline(y=target_angle, color='r', linestyle='--', label='Target Angle')
    plt.title(f'Position Trajectory of moving from {current_angle} to {target_angle} for actuator {actuator_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.grid(True)
    plt.legend()

    # Plot velocity trajectory
    plt.subplot(2, 1, 2)
    plt.plot(times[:-1], velocities, label=f"{profile.capitalize()} Profile")
    plt.title(f'Velocity Profile of moving from {current_angle} to {target_angle} for actuator {actuator_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (deg/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/{current_angle}_to_{target_angle}_for_{actuator_id}_{profile}_{timestamp}.png')
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Print statistics if requested
    # if print_stats:
    #     print(f"\n{profile.capitalize()} Profile Statistics:")
    #     print(f"  Total time: {times[-1]:.3f} seconds")
    #     print(f"  Max velocity: {max(abs(v) for v in velocities):.3f} deg/s")
    #     print(f"  Final position: {angles[-1]:.3f} deg")
    
    return trajectory_data


# Example usage (will only run if script is executed directly)
if __name__ == "__main__":
    # Example with default parameters
    # plot_motion_plan()
    
    # Example with custom parameters
    plot_motion_plan(
        current_angle=0.0,
        target_angle=10,
        acceleration=300,
        v_max=40,
        update_rate=100,
        profile="scurve",
        actuator_id=44
    )
