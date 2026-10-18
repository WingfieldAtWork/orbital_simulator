# orbit.py

import numpy as np
import matplotlib.pyplot as plt

# Constants
mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)

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
    
    # Calculate acceleration vector
    a = -(mu / r_mag_cubed) * r
    
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
    # Create the figure and a 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 3. Slice the columns for X, Y, and Z and plot
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', marker='o')

    # Set labels for all three axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

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
    steps = 50

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