# orbit.py

import numpy as np
import matplotlib.pyplot as plt

# Constants
mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
re = 6378137 # Earth's equatorial radius
j2 = 1.08263e-3 # Oblateness constant of Earth

def calculate_peturbation(state, r_mag_sq):
    # Couple of variables for readability
    r_mag = np.sqrt(r_mag_sq)
    x = state[0]
    y = state[1]
    z = state[2]

    # Calculate each component of the peturbation acceleration
    aj2_const = -1.5 * j2 * mu/r_mag_sq * (re/r_mag)**2 * (1 - 5*z/r_mag_sq)
    aj2_const_z = -1.5 * j2 * mu/r_mag_sq * (re/r_mag)**2 * (3 - 5*z/r_mag_sq)
    aj2 = np.array([aj2_const * x/r_mag, aj2_const * y/r_mag, aj2_const_z * z/r_mag])

    return aj2


def gravity_derivative(state):
    """
    Calculates the derivative of the 6D state vector dX/dt = [v, a].
    state: numpy array of shape (6,) -> [x, y, z, vx, vy, vz]
    """
    # Split position and velocity using slicing
    r = state[0:3]
    v = state[3:6]
    
    # Calculate distance cubed
    # np.dot(r, r) is faster than np.linalg.norm(r)**2
    r_mag_sq = np.dot(r, r)
    r_mag_cubed = r_mag_sq * np.sqrt(r_mag_sq)
    
    # Calculate acceleration vector with J2 peturbation
    a = -(mu / r_mag_cubed) * r
    aj2 = calculate_peturbation(state, r_mag_sq)
    a = a + aj2
    
    # Concatenate velocity and acceleration back into a 6D derivative array
    return np.concatenate((v, a))


def rk4_step(state, dt):
    # RK4 uses 4 midpoints, with the two middle weighted
    k1 = gravity_derivative(state)
    k2 = gravity_derivative(state + 0.5 * dt * k1)
    k3 = gravity_derivative(state + 0.5 * dt * k2)
    k4 = gravity_derivative(state + dt * k3)
    
    # Weighted average update step
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_orbit(initial_state, dt, total_steps):
    # Pre-allocate array to store results: shape (total_steps, 6)
    history = np.zeros((total_steps, 6))

    # Calculate the current state
    current_state = np.array(initial_state, dtype=float)
    history[0] = current_state
    
    # For each step, calculate the current state vector and add it to the history
    for step in range(1, total_steps):
        current_state = rk4_step(current_state, dt)
        history[step] = current_state
        
    return history


def plot_orbit(positions):
    # 1. Initialize the figure
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    units = 'm'

    # 2. Generate a parametric sphere mesh
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x_earth = re * np.outer(np.cos(u), np.sin(v))
    y_earth = re * np.outer(np.sin(u), np.sin(v))
    z_earth = re * np.outer(np.ones_like(u), np.cos(v))

    # 3. Plot Earth surface (semi-transparent so occulted orbits remain visible)
    ax.plot_surface(
        x_earth, y_earth, z_earth,
        color='royalblue',
        alpha=0.35,
        edgecolor='navy',
        linewidth=0.2,
        rstride=2,
        cstride=2
    )

    # 4. Plot satellite trajectory with a line plot
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color='crimson', linewidth=1.5, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='green', s=40, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='red', s=40, label='Current')

    # 5. Lock aspect ratio so Earth remains a sphere instead of an ellipsoid
    ax.set_box_aspect([1, 1, 1])

    # 6. Ensure axis limits accommodate both Earth and the full orbit envelope
    max_val = max(np.max(np.abs(positions)), re * 1.1)
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    unit_label = 'km' if units == 'km' else 'm'
    ax.set_xlabel(f'ECI X ({unit_label})')
    ax.set_ylabel(f'ECI Y ({unit_label})')
    ax.set_zlabel(f'ECI Z ({unit_label})')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


"""
The calculation loop follows as:
- Given initial r and v at t = 0, find the magnitude of r_0
- Calculate the scalar strength of gravity, g_strength = mu/r_0_magnitude
- Find initial acceleration, a_0 = g_strength * r_0
- Now, numerically integrate to calculate the next step:
    - delta_t = 10
    - r_1 = r_0 + v_0*delta_t
    - v_1 = v_0 + a_0*delta_t
- Generalized, you repeat this using each previous step
"""
def main():
    # Initial State Vector (Position in m, Velocity in m/s)
    # A typical Low Earth Orbit (LEO)
    r0 = [6771000.0, 0.0, 0.0] # r_x, r_y, r_z
    v0 = [0.0, 7670.0, 0.0]   # v_x, v_y, v_z
    x0 = r0 + v0

    # Simulation steps
    dt = 10
    steps = 600

    # Run calculations
    orbit_history = propagate_orbit(x0, dt, steps)

    # Verify the results at the final step
    final_state = orbit_history[-1]
    print("Final Position (m):", final_state[0:3])
    print("Final Velocity (m/s):", final_state[3:6])
    print(orbit_history)

    # Plot the position of the object over each step
    # (Yes, I know this is slightly inefficient)
    plot_orbit(orbit_history[:, :3])


if __name__ == "__main__":
    main()