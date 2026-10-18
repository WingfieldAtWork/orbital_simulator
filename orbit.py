# orbit.py

import numpy as np

# Constants
mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)

# Initial State Vector (Position in m, Velocity in m/s)
# A typical Low Earth Orbit (LEO)
r0 = np.array([6771000.0, 0.0, 0.0]) # r_x, r_y, r_z
v0 = np.array([0.0, 7670.0, 0.0])   # v_x, v_y, v_z


# Euler numerical integration for calculating the next step
# x_n+1 = x_n + diff(x_n, t)*dt
def euler(r_prev, v_prev, a, dt):
    r = r_prev + v_prev*dt
    v = v_prev + a*dt
    return r, v

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
    # Initialize the loop
    steps = 10
    dt = 10
    r = r0
    v = v0
    a = 0

    # Begin calculation loop
    for _ in range(steps):
        # Calculate the magnitude
        r_mag = np.linalg.norm(r)

        # Calculate acceleration
        a = -(mu / (r_mag)**3) * r0

        # Calculate the next step
        r_prev = r
        v_prev = v

        r, v = euler(r_prev, v_prev, a, dt)

        # Print results
        print(f"r_mag: {r_mag}")
        print(f"a: {a}")
        print(f"r: {r}")
        print(f"v: {v}\n")

if __name__ == "__main__":
    main()