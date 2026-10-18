"""
estimator.py - Flight Software Extended Kalman Filter (EKF)

Implements real-time onboard orbit determination using an Extended Kalman Filter
(EKF). Propagates state and covariance through the non-linear orbital dynamics
model and executes discrete measurement updates using incoming GPS position fixes.

Rule (from specs.md):
- Must NEVER import dynamics.py.
- Only sees what real flight software would see: elapsed time dt and incoming
  sensor measurement packets z_k with covariance R.
- Maintains its own internal physics model, Jacobian derivation, and numerical integrator.
"""

from typing import Optional
import numpy as np

from config import MU_EARTH, R_EARTH, J2_EARTH


class ExtendedKalmanFilter:
    """
    Flight Software Extended Kalman Filter for real-time 6D orbital state estimation.

    State vector:
        x = [x, y, z, vx, vy, vz]^T (position in m, velocity in m/s)
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_cov: Optional[np.ndarray] = None,
        process_noise_diag: Optional[np.ndarray] = None,
        mu: float = MU_EARTH,
        j2: float = J2_EARTH,
        r_earth: float = R_EARTH,
    ) -> None:
        """
        Initializes the flight software EKF.

        Parameters:
            initial_state: 6D initial state guess [x0, y0, z0, vx0, vy0, vz0].
            initial_cov: Optional 6x6 initial error covariance matrix P0.
                         If None, defaults to diag([100^2, 100^2, 100^2, 5^2, 5^2, 5^2]).
            process_noise_diag: Optional 6D array specifying continuous process noise
                                power spectral density Q_c.
            mu: Central body gravitational parameter (m^3 / s^2).
            j2: Second zonal harmonic coefficient (oblateness).
            r_earth: Central body equatorial radius (meters).
        """
        self._state = np.array(initial_state, dtype=float).copy()
        if self._state.shape != (6,):
            raise ValueError(f"initial_state must have shape (6,), got {self._state.shape}")

        # Initialize error covariance P (6x6)
        if initial_cov is not None:
            self._P = np.array(initial_cov, dtype=float).copy()
        else:
            # Default: 100 m 1-sigma in position, 5 m/s 1-sigma in velocity
            p_pos = 100.0 ** 2
            p_vel = 5.0 ** 2
            self._P = np.diag([p_pos, p_pos, p_pos, p_vel, p_vel, p_vel]).astype(float)

        # Process noise continuous spectral density Q_c (6x6)
        if process_noise_diag is not None:
            self._Q_c = np.diag(process_noise_diag).astype(float)
        else:
            # Moderate process noise to account for higher-order gravity, atmospheric drag, and unmodeled perturbations
            q_pos = 1e-4  # m^2 / s
            q_vel = 1e-3  # (m/s)^2 / s
            self._Q_c = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(float)

        # Flight software internal gravity constants
        self.mu = float(mu)
        self.j2 = float(j2)
        self.r_earth = float(r_earth)

        # Measurement mapping matrix H: z = [x, y, z] -> H = [I_3, 0_3x3]
        self._H = np.zeros((3, 6), dtype=float)
        self._H[0:3, 0:3] = np.eye(3, dtype=float)

    @property
    def state(self) -> np.ndarray:
        """Current estimated 6D state vector [x, y, z, vx, vy, vz]."""
        return self._state.copy()

    @property
    def P(self) -> np.ndarray:
        """Current 6x6 estimation error covariance matrix."""
        return self._P.copy()

    @property
    def position_cov(self) -> np.ndarray:
        """3x3 position error covariance submatrix (m^2)."""
        return self._P[0:3, 0:3].copy()

    @property
    def velocity_cov(self) -> np.ndarray:
        """3x3 velocity error covariance submatrix ((m/s)^2)."""
        return self._P[3:6, 3:6].copy()

    @property
    def three_sigma(self) -> np.ndarray:
        """Theoretical +-3-sigma estimation uncertainty bounds (6D vector)."""
        return 3.0 * np.sqrt(np.maximum(np.diag(self._P), 0.0))

    def _fsw_acceleration(self, r: np.ndarray) -> np.ndarray:
        """
        Internal flight software gravity model: Keplerian central force + J2 oblateness.
        """
        x, y, z = r[0], r[1], r[2]
        r_sq = float(np.dot(r, r))
        r_mag = np.sqrt(r_sq)

        # Keplerian acceleration: a_kepler = -mu * r / r^3
        a_kepler = -(self.mu / (r_sq * r_mag)) * r

        # J2 harmonic perturbation
        if self.j2 != 0.0:
            factor = 1.5 * self.j2 * self.mu * (self.r_earth**2) / (r_sq * r_sq * r_mag)
            z2_over_r2 = 5.0 * (z**2) / r_sq
            ax_j2 = -factor * x * (1.0 - z2_over_r2)
            ay_j2 = -factor * y * (1.0 - z2_over_r2)
            az_j2 = -factor * z * (3.0 - z2_over_r2)
            a_j2 = np.array([ax_j2, ay_j2, az_j2], dtype=float)
        else:
            a_j2 = np.zeros(3, dtype=float)

        return a_kepler + a_j2

    def _fsw_dynamics(self, state: np.ndarray) -> np.ndarray:
        """
        Time derivative of the state vector: dX/dt = [v, a].
        """
        v = state[3:6]
        a = self._fsw_acceleration(state[0:3])
        return np.concatenate((v, a))

    def _compute_state_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the continuous-time dynamics Jacobian matrix F_c = df/dx:
            F_c = [ 0_3x3   I_3x3 ]
                  [ G(r)    0_3x3 ]
        where G(r) = da/dr is the 3x3 gravity gradient tensor.
        """
        r = state[0:3]
        x, y, z = r[0], r[1], r[2]
        r_sq = float(np.dot(r, r))
        r_mag = np.sqrt(r_sq)

        # 1. Keplerian gravity gradient: G_kepler = -(mu/r^3) * I + (3*mu/r^5) * r * r^T
        G_kepler = -(self.mu / (r_sq * r_mag)) * np.eye(3, dtype=float) + (
            (3.0 * self.mu / (r_sq * r_sq * r_mag)) * np.outer(r, r)
        )

        # 2. J2 gravity gradient
        G_j2 = np.zeros((3, 3), dtype=float)
        if self.j2 != 0.0:
            C = 1.5 * self.j2 * self.mu * (self.r_earth**2)
            r5 = r_sq * r_sq * r_mag
            r7 = r5 * r_sq
            r9 = r7 * r_sq
            z2 = z**2

            # Partial derivatives da_j2 / dr
            G_j2[0, 0] = -C * ((1.0 / r5 - 5.0 * z2 / r7) + x * (-5.0 * x / r7 + 35.0 * x * z2 / r9))
            G_j2[0, 1] = -C * (x * (-5.0 * y / r7 + 35.0 * y * z2 / r9))
            G_j2[0, 2] = -C * (-10.0 * x * z / r7 + x * (-5.0 * z / r7 + 35.0 * (z**3) / r9))

            G_j2[1, 0] = G_j2[0, 1]
            G_j2[1, 1] = -C * ((1.0 / r5 - 5.0 * z2 / r7) + y * (-5.0 * y / r7 + 35.0 * y * z2 / r9))
            G_j2[1, 2] = -C * (-10.0 * y * z / r7 + y * (-5.0 * z / r7 + 35.0 * (z**3) / r9))

            G_j2[2, 0] = G_j2[0, 2]
            G_j2[2, 1] = G_j2[1, 2]
            G_j2[2, 2] = -C * ((3.0 / r5 - 25.0 * z2 / r7) + z * (-5.0 * z / r7 + 35.0 * (z**3) / r9))

        G_total = G_kepler + G_j2

        # Assemble 6x6 continuous Jacobian F_c
        F = np.zeros((6, 6), dtype=float)
        F[0:3, 3:6] = np.eye(3, dtype=float)
        F[3:6, 0:3] = G_total

        return F

    def predict(self, dt: float) -> None:
        """
        Propagates state estimate and covariance forward by time step dt.

        Parameters:
            dt: Time step duration (seconds).
        """
        # A. Non-linear state propagation via internal RK4 integrator
        k1 = self._fsw_dynamics(self._state)
        k2 = self._fsw_dynamics(self._state + 0.5 * dt * k1)
        k3 = self._fsw_dynamics(self._state + 0.5 * dt * k2)
        k4 = self._fsw_dynamics(self._state + dt * k3)
        self._state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # B. Discrete-time state transition matrix Phi = exp(F_c * dt) via 2nd-order Taylor expansion
        F = self._compute_state_jacobian(self._state)
        F_dt = F * dt
        Phi = np.eye(6, dtype=float) + F_dt + 0.5 * (F_dt @ F_dt)

        # C. Discrete process noise covariance Q_d = Q_c * dt
        Q_d = self._Q_c * dt

        # D. Covariance propagation: P_k|k-1 = Phi * P_k-1|k-1 * Phi^T + Q_d
        self._P = (Phi @ self._P @ Phi.T) + Q_d

        # Symmetrize to prevent numerical asymmetry accumulation
        self._P = 0.5 * (self._P + self._P.T)

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        """
        Executes Kalman measurement update with incoming GPS position fix z and covariance R.

        Parameters:
            z: 3D measured position vector [x_meas, y_meas, z_meas] in meters.
            R: 3x3 sensor measurement error covariance matrix (m^2).
        """
        z_meas = np.asarray(z, dtype=float).reshape(3)
        R_mat = np.asarray(R, dtype=float).reshape(3, 3)

        # 1. Innovation (measurement residual): y_tilde = z - h(x_pred)
        # Measurement model: h(x) = x[0:3]
        pred_pos = self._state[0:3]
        y_tilde = z_meas - pred_pos

        # 2. Innovation covariance: S = H * P * H^T + R
        # Since H = [I_3, 0_3], H * P * H^T is simply the upper-left 3x3 position covariance
        P_pos = self._P[0:3, 0:3]
        S = P_pos + R_mat

        # 3. Kalman Gain: K = P * H^T * S^-1
        # P * H^T is the 6x3 matrix formed by the first three columns of P
        P_HT = self._P[:, 0:3]
        # Use solve for numerical robustness: K = (S^-1 * P_HT^T)^T
        K = np.linalg.solve(S, P_HT.T).T  # Shape: (6, 3)

        # 4. State correction: x = x + K * y_tilde
        self._state += K @ y_tilde

        # 5. Joseph stabilized covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        # Guarantees that P remains symmetric and positive semi-definite
        I_KH = np.eye(6, dtype=float) - (K @ self._H)
        self._P = (I_KH @ self._P @ I_KH.T) + (K @ R_mat @ K.T)

        # Enforce exact symmetry
        self._P = 0.5 * (self._P + self._P.T)

