"""
sensors.py - Hardware Emulation & Sensor Noise Injection

Emulates discrete, multi-rate spaceflight navigation hardware:
- GPSSensor: Generates noisy 3D position fixes at a configured polling rate.
- IMUSensor: Generates noisy specific force / angular rate measurements.

Rule (from specs.md):
- Manages its own internal clock / sample rate.
- Manages its own independent pseudo-random number generator.
- Completely decoupled from estimator and dynamics.
"""

from typing import Optional, Tuple
import numpy as np


class GPSSensor:
    """
    Emulates a spaceborne Global Positioning System (GPS) receiver.

    Produces discrete 3D position measurement fixes corrupted by zero-mean
    Gaussian noise at a specified polling rate (e.g. 1 Hz).
    """

    def __init__(
        self,
        sigma_pos: float = 5.0,
        sample_rate_hz: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the GPS sensor.

        Parameters:
            sigma_pos: 1-sigma standard deviation of position error per axis (meters).
            sample_rate_hz: Hardware polling rate in Hertz (e.g. 1.0 Hz).
            seed: Optional random number generator seed for deterministic repeatability.
        """
        self.sigma_pos = float(sigma_pos)
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_interval = 1.0 / self.sample_rate_hz if self.sample_rate_hz > 0 else float("inf")

        # Independent random number generator (specs rule)
        self._rng = np.random.default_rng(seed)

        # Calibrated measurement noise covariance matrix R (3x3)
        self._R = (self.sigma_pos ** 2) * np.eye(3, dtype=float)

        # Internal clock tracker
        self._last_sample_time: Optional[float] = None

    @property
    def R(self) -> np.ndarray:
        """Calibrated measurement noise covariance matrix (3x3, m^2)."""
        return self._R.copy()

    @property
    def sample_rate(self) -> float:
        """Polling rate in Hertz."""
        return self.sample_rate_hz

    def poll(self, current_time: float, true_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Polls the GPS hardware at simulation timestamp current_time.

        Parameters:
            current_time: Current simulation timestamp in seconds.
            true_pos: Ground truth 3D position vector [x, y, z] in meters.

        Returns:
            np.ndarray: Corrupted 3D position measurement vector [x_m, y_m, z_m]
                        if hardware sampling period has elapsed; None otherwise.
        """
        if self._last_sample_time is None:
            # First fix available at start of mission
            self._last_sample_time = current_time
            noise = self._rng.normal(0.0, self.sigma_pos, size=3)
            return np.asarray(true_pos, dtype=float) + noise

        # Check if hardware sampling interval has elapsed (accounting for floating point precision)
        dt_elapsed = current_time - self._last_sample_time
        if dt_elapsed >= self.sample_interval - 1e-9:
            self._last_sample_time = current_time
            noise = self._rng.normal(0.0, self.sigma_pos, size=3)
            return np.asarray(true_pos, dtype=float) + noise

        return None

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the internal hardware clock and optional RNG seed."""
        self._last_sample_time = None
        if seed is not None:
            self._rng = np.random.default_rng(seed)


class IMUSensor:
    """
    Emulates a strapdown Inertial Measurement Unit (IMU) with high-rate accelerometer
    and rate gyroscope measurements corrupted by Gaussian noise and random walk bias.
    """

    def __init__(
        self,
        accel_noise_density: float = 0.01,
        gyro_noise_density: float = 0.001,
        sample_rate_hz: float = 50.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the IMU sensor.

        Parameters:
            accel_noise_density: Accelerometer 1-sigma white noise (m/s^2).
            gyro_noise_density: Gyroscope 1-sigma white noise (rad/s).
            sample_rate_hz: Sampling frequency in Hertz.
            seed: Optional RNG seed.
        """
        self.accel_noise = float(accel_noise_density)
        self.gyro_noise = float(gyro_noise_density)
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_interval = 1.0 / self.sample_rate_hz if self.sample_rate_hz > 0 else float("inf")

        self._rng = np.random.default_rng(seed)
        self._last_sample_time: Optional[float] = None

    def poll(
        self,
        current_time: float,
        true_accel: np.ndarray,
        true_omega: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Polls IMU for specific force (acceleration) and angular velocity.
        """
        if self._last_sample_time is None or (current_time - self._last_sample_time) >= self.sample_interval - 1e-9:
            self._last_sample_time = current_time
            accel_meas = np.asarray(true_accel, dtype=float) + self._rng.normal(0.0, self.accel_noise, size=3)
            omega_true = np.zeros(3) if true_omega is None else true_omega
            omega_meas = np.asarray(omega_true, dtype=float) + self._rng.normal(0.0, self.gyro_noise, size=3)
            return accel_meas, omega_meas

        return None

