"""
config.py - Simulation Parameters and Physical Constants

Centralizes all orbital mechanics constants, simulation timing parameters,
and default initial states for truth dynamics and flight software estimators.
"""

from dataclasses import dataclass, field
import numpy as np

# ==============================================================================
# Physical Constants
# ==============================================================================
MU_EARTH: float = 3.986004418e14    # Earth's standard gravitational parameter (m^3 / s^2)
R_EARTH: float = 6378137.0          # Earth's equatorial radius (meters)
J2_EARTH: float = 1.08263e-3        # Earth's second gravitational zonal harmonic (oblateness)
OMEGA_EARTH: float = 7.292115e-5    # Earth's mean angular rotation rate (rad / s)
SPEED_OF_LIGHT: float = 299792458.0 # Speed of light in vacuum (m / s)


def create_circular_leo_state(
    altitude_m: float = 392863.0,
    inclination_deg: float = 28.5,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
) -> np.ndarray:
    """
    Generates a 6D Cartesian orbital state vector [x, y, z, vx, vy, vz]
    for an inclined circular Low Earth Orbit (LEO).

    Parameters:
        altitude_m: Orbital altitude above Earth's equatorial surface (m).
                    Default (~392.86 km) yields orbital radius r = 6,771,000 m.
        inclination_deg: Orbital inclination relative to the equatorial plane (deg).
                         Default 28.5 deg corresponds to a Cape Canaveral launch.
        mu: Gravitational parameter (m^3 / s^2).
        r_earth: Equatorial radius (m).

    Returns:
        np.ndarray: 6D state vector [x, y, z, vx, vy, vz] in meters and m/s.
    """
    # Calculate the radius from the earth to the satellite
    r_orbit = r_earth + altitude_m
    # Velocity in orbit
    v_orbit = np.sqrt(mu / r_orbit)
    # Inclination in radians
    inc_rad = np.radians(inclination_deg)

    # Initial position along X axis
    r0 = np.array([r_orbit, 0.0, 0.0], dtype=float)

    # Initial velocity vector inclined in Y-Z plane
    v0 = np.array([0.0, v_orbit * np.cos(inc_rad), v_orbit * np.sin(inc_rad)], dtype=float)

    # Return state vector
    return np.concatenate((r0, v0))


@dataclass
class SimConfig:
    """
    Configuration dataclass centralizing physical constants, timing parameters,
    and initial state vectors for the orbital simulation.
    """
    # Gravitational constants
    MU: float = MU_EARTH
    R_EARTH: float = R_EARTH
    J2: float = J2_EARTH

    # Simulation timeline parameters (specs.md: 50 Hz physics, 1 Hz GPS)
    DT_PHYSICS: float = 0.02       # Physics integration step dt_sim (s) -> 50 Hz
    DURATION: float = 5550.0       # Total duration (s) ~1 orbit period (~92.5 min)
    GPS_RATE: float = 1.0          # GPS sensor polling rate (Hz) -> dt_gps = 1.0 s
    GPS_SIGMA: float = 5.0         # 1-sigma GPS position measurement noise (meters)

    # Default initial orbit parameters
    ORBIT_ALTITUDE_M: float = 392863.0  # Gives r0 = 6,771,000 m
    ORBIT_INCLINATION_DEG: float = 28.5 # Cape Canaveral inclination

    # Ground truth initial state [x, y, z, vx, vy, vz]
    TRUE_INITIAL_STATE: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # Initial guess for the flight software / EKF filter
    INITIAL_STATE_GUESS: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # Initial estimation covariance diagonal [pos_var (m^2), vel_var (m/s)^2]
    INITIAL_COV_DIAG: np.ndarray = field(
        default_factory=lambda: np.array([100.0**2, 100.0**2, 100.0**2, 5.0**2, 5.0**2, 5.0**2])
    )

    # Process noise covariance diagonal for EKF [pos_noise (m^2/s), vel_noise (m^2/s^3)]
    PROCESS_NOISE_Q_DIAG: np.ndarray = field(
        default_factory=lambda: np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
    )

    def __post_init__(self):
        # Initialize true state if not explicitly overridden
        if np.all(self.TRUE_INITIAL_STATE == 0):
            self.TRUE_INITIAL_STATE = create_circular_leo_state(
                altitude_m=self.ORBIT_ALTITUDE_M,
                inclination_deg=self.ORBIT_INCLINATION_DEG,
                mu=self.MU,
                r_earth=self.R_EARTH,
            )

        # Initialize filter initial guess with realistic offset if not provided
        if np.all(self.INITIAL_STATE_GUESS == 0):
            # Realistic initial offset: ~50m in position, ~1 m/s in velocity
            pos_offset = np.array([50.0, -30.0, 20.0])
            vel_offset = np.array([1.0, -0.5, 0.8])
            self.INITIAL_STATE_GUESS = self.TRUE_INITIAL_STATE.copy()
            self.INITIAL_STATE_GUESS[:3] += pos_offset
            self.INITIAL_STATE_GUESS[3:] += vel_offset

    @property
    def DT_GPS(self) -> float:
        """GPS sampling period in seconds."""
        return 1.0 / self.GPS_RATE if self.GPS_RATE > 0 else float("inf")

