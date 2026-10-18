"""
dynamics.py - Handles calculations of orbit for base comparison to sensors

Computes deterministic orbital equations of motion including two-body
gravity and Earth's J2 oblateness perturbation, calculated forward
in time via a 4th-order Runge-Kutta (RK4) integration.
"""

from typing import Union
import numpy as np

from config import MU_EARTH, R_EARTH, J2_EARTH


class OrbitalDynamics:
    """
    Deterministic equations of motion and numerical integrator for orbital propagation.
    """

    def __init__(
        self,
        mu: float = MU_EARTH,
        j2: float = J2_EARTH,
        r_earth: float = R_EARTH,
    ) -> None:
        """
        Initializes the orbital dynamics engine.

        Parameters:
            mu: Gravitational parameter of the central body (m^3 / s^2).
            j2: Second zonal harmonic coefficient (dimensionless oblateness).
            r_earth: Equatorial radius of the central body (meters).
        """
        self.mu = float(mu)
        self.j2 = float(j2)
        self.r_earth = float(r_earth)

    def compute_j2_acceleration(self, r: np.ndarray) -> np.ndarray:
        """
        Calculates the gravitational perturbation acceleration due to Earth's
        oblateness (J2 harmonic) in Cartesian ECI coordinates.

        Formula (WGS-84 / Vallado astrodynamics):
            factor = (3/2) * J2 * mu * (R_E^2) / (r^5)
            z_ratio = 5 * (z^2 / r^2)
            a_j2_x = -factor * x * (1 - z_ratio)
            a_j2_y = -factor * y * (1 - z_ratio)
            a_j2_z = -factor * z * (3 - z_ratio)
        """
        if self.j2 == 0.0:
            return np.zeros(3, dtype=float)

        x, y, z = r[0], r[1], r[2]
        r_sq = float(np.dot(r, r))
        r_mag = np.sqrt(r_sq)

        # Factor = 1.5 * J2 * mu * R_E^2 / r^5
        factor = 1.5 * self.j2 * self.mu * (self.r_earth**2) / (r_sq * r_sq * r_mag)
        z2_over_r2 = 5.0 * (z**2) / r_sq

        ax = -factor * x * (1.0 - z2_over_r2)
        ay = -factor * y * (1.0 - z2_over_r2)
        az = -factor * z * (3.0 - z2_over_r2)

        return np.array([ax, ay, az], dtype=float)

    def compute_derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the time derivative of the 6D Cartesian orbital state vector:
            dX/dt = [v, a] = [vx, vy, vz, ax, ay, az]

        Parameters:
            state: 6D numpy array [x, y, z, vx, vy, vz] in meters and m/s.

        Returns:
            np.ndarray: 6D derivative vector [vx, vy, vz, ax, ay, az].
        """
        r = state[0:3]
        v = state[3:6]

        r_sq = float(np.dot(r, r))
        r_mag = np.sqrt(r_sq)

        # Central two-body Keplerian gravity: a = -mu * r / r^3
        a_kepler = -(self.mu / (r_sq * r_mag)) * r

        # J2 oblateness perturbation
        a_j2 = self.compute_j2_acceleration(r)

        total_a = a_kepler + a_j2

        return np.concatenate((v, total_a))

    def gravity_derivative(self, state: np.ndarray) -> np.ndarray:
        """Alias for compute_derivatives for compatibility with orbit.py."""
        return self.compute_derivatives(state)

    def step_rk4(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagates the 6D state vector forward by time step dt using
        the classical 4th-order Runge-Kutta (RK4) integration algorithm.

        Parameters:
            state: Current 6D state vector [x, y, z, vx, vy, vz].
            dt: Integration time step (seconds).

        Returns:
            np.ndarray: Propagated state vector at time t + dt.
        """
        k1 = self.compute_derivatives(state)
        k2 = self.compute_derivatives(state + 0.5 * dt * k1)
        k3 = self.compute_derivatives(state + 0.5 * dt * k2)
        k4 = self.compute_derivatives(state + dt * k3)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate(
        self,
        initial_state: Union[list, np.ndarray],
        dt: float,
        total_steps: int,
    ) -> np.ndarray:
        """
        Propagates an initial state forward across total_steps intervals of dt.

        Parameters:
            initial_state: Initial 6D state vector [x0, y0, z0, vx0, vy0, vz0].
            dt: Time step per step (seconds).
            total_steps: Number of integration steps to perform.

        Returns:
            np.ndarray: Trajectory history of shape (total_steps, 6).
        """
        history = np.zeros((total_steps, 6), dtype=float)
        current_state = np.array(initial_state, dtype=float)
        history[0] = current_state

        for step in range(1, total_steps):
            current_state = self.step_rk4(current_state, dt)
            history[step] = current_state

        return history

    def specific_energy(self, state: np.ndarray) -> float:
        """
        Calculates specific mechanical energy (eps = v^2/2 - mu/r + Phi_J2) (J/kg).
        Useful for checking numerical energy conservation.
        """
        r = state[:3]
        v = state[3:]
        r_mag = np.linalg.norm(r)
        v_sq = float(np.dot(v, v))

        # Two-body potential: -mu/r
        kepler_potential = -self.mu / r_mag

        # J2 potential perturbation
        z_over_r = r[2] / r_mag
        j2_potential = (
            (self.mu * self.j2 * self.r_earth**2)
            / (2.0 * r_mag**3)
            * (3.0 * z_over_r**2 - 1.0)
        )

        return 0.5 * v_sq + kepler_potential + j2_potential

    def altitude(self, state: np.ndarray) -> float:
        """Calculates instantaneous altitude above equatorial radius (meters)."""
        return float(np.linalg.norm(state[:3]) - self.r_earth)

    def angular_momentum(self, state: np.ndarray) -> np.ndarray:
        """Calculates specific angular momentum vector h = r x v (m^2/s)."""
        return np.cross(state[:3], state[3:])

