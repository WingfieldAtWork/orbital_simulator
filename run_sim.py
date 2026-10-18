"""
run_sim.py - SImulation Coordinator

Coordinates the simulation clock loop, passing messages between files:
- dynamics.py (does base physics calculations and sends back data)
- sensors.py (discrete hardware emulation)
- estimator.py (discrete state estimation with Extended Kalman Filter)
- visualizer.py (take telemetry and plots history)
"""

import sys
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np

from config import SimConfig
from dynamics import OrbitalDynamics
from visualizer import plot_orbit_and_residuals

# Optional imports for modules to be implemented for future implementations
try:
    from sensors import GPSSensor
except ImportError:
    GPSSensor = None

try:
    from estimator import ExtendedKalmanFilter
except ImportError:
    ExtendedKalmanFilter = None


def run_simulation(
    cfg: Optional[SimConfig] = None,
    show_progress: bool = True,
    progress_interval_s: float = 1000.0,
) -> Dict[str, Any]:
    """
    Executes the master orbital simulation clock loop.

    Parameters:
        cfg: SimConfig configuration instance. If None, default SimConfig() is used.
        show_progress: Whether to print simulation progress to console.
        progress_interval_s: Sim time interval (s) at which to print progress updates.

    Returns:
        dict: Telemetry logs containing 'time', 'truth', 'meas', 'est', and 'cov'.
    """
    if cfg is None:
        cfg = SimConfig()

    print("=" * 70)
    print("ORBITAL SIMULATOR - MISSION COORDINATOR")
    print("=" * 70)
    print(f"Physics Timestep (DT_PHYSICS): {cfg.DT_PHYSICS:.4f} s ({(1.0 / cfg.DT_PHYSICS):.1f} Hz)")
    print(f"Simulation Duration:          {cfg.DURATION:.1f} s (~{(cfg.DURATION / 60.0):.1f} min)")
    print(f"Earth Gravitational Param:    {cfg.MU:.6e} m^3/s^2")
    print(f"J2 Oblateness Perturbation:   {cfg.J2:.5e}")

    # 1. Instantiate truth dynamics engine
    truth_engine = OrbitalDynamics(mu=cfg.MU, j2=cfg.J2, r_earth=cfg.R_EARTH)
    print("Subsystem [Dynamics]:         ACTIVE (Ground Truth RK4 Engine)")

    # 2. Check and instantiate hardware emulation & flight software
    has_sensors = GPSSensor is not None
    has_estimator = ExtendedKalmanFilter is not None

    gps = None
    ekf = None

    if has_sensors:
        gps = GPSSensor(sigma_pos=cfg.GPS_SIGMA, sample_rate_hz=cfg.GPS_RATE)
        print(f"Subsystem [Sensors]:          ACTIVE (GPS @ {cfg.GPS_RATE} Hz, sigma={cfg.GPS_SIGMA}m)")
    else:
        print("Subsystem [Sensors]:          PENDING (sensors.py not detected - skipping hardware polling)")

    if has_estimator:
        ekf = ExtendedKalmanFilter(initial_state=cfg.INITIAL_STATE_GUESS)
        print("Subsystem [Estimator]:        ACTIVE (Extended Kalman Filter Flight Software)")
    else:
        print("Subsystem [Estimator]:        PENDING (estimator.py not detected - skipping state filtering)")

    # 3. Initialize telemetry storage
    history: Dict[str, Any] = {
        "time": [],
        "truth": [],
        "meas": [],
        "est": [],
        "cov": [],
    }

    current_state = cfg.TRUE_INITIAL_STATE.copy()
    start_wall_time = time.time()
    last_reported_sim_t = -progress_interval_s

    # Generate simulation time grid
    time_steps = np.arange(0.0, cfg.DURATION, cfg.DT_PHYSICS)
    total_steps = len(time_steps)

    print("-" * 70)
    print(f"Running simulation loop across {total_steps:,} steps...")

    # 4. Master Simulation Clock Loop
    for step_idx, t in enumerate(time_steps):
        # A. Advance physical reality (Ground truth propagation)
        current_state = truth_engine.step_rk4(current_state, cfg.DT_PHYSICS)

        # B. Propagate EKF state prediction (internal physics model)
        if ekf is not None:
            ekf.predict(dt=cfg.DT_PHYSICS)

        # C. Poll sensor hardware
        measurement = None
        if gps is not None:
            measurement = gps.poll(current_time=t, true_pos=current_state[:3])

        # D. If sensor fix arrives, execute measurement update
        if measurement is not None and ekf is not None:
            ekf.update(z=measurement, R=gps.R)
            history["meas"].append((t, measurement))

        # E. Record telemetry
        history["time"].append(t)
        history["truth"].append(current_state.copy())

        if ekf is not None:
            history["est"].append(ekf.state.copy())
            history["cov"].append(np.diag(ekf.P).copy())

        # Progress reporting
        if show_progress and (t - last_reported_sim_t >= progress_interval_s or step_idx == total_steps - 1):
            pct = 100.0 * (step_idx + 1) / total_steps
            alt_km = truth_engine.altitude(current_state) / 1000.0
            spd_km_s = np.linalg.norm(current_state[3:6]) / 1000.0
            print(
                f"  [Sim T = {t:6.1f}s / {cfg.DURATION:6.1f}s ({pct:5.1f}%)] "
                f"Altitude: {alt_km:7.2f} km | Speed: {spd_km_s:6.3f} km/s"
            )
            last_reported_sim_t = t

    wall_duration = time.time() - start_wall_time
    sim_speedup = cfg.DURATION / wall_duration if wall_duration > 0 else float("inf")

    # Convert truth list to numpy array for efficient downstream handling
    history["time"] = np.array(history["time"])
    history["truth"] = np.array(history["truth"])
    if len(history["est"]) > 0:
        history["est"] = np.array(history["est"])
    if len(history["cov"]) > 0:
        history["cov"] = np.array(history["cov"])

    print("-" * 70)
    print(f"Simulation completed in {wall_duration:.2f} seconds ({sim_speedup:.1f}x real-time).")
    final_pos = current_state[:3]
    final_vel = current_state[3:6]
    final_alt = truth_engine.altitude(current_state) / 1000.0
    print(f"Final Position (km): [{final_pos[0]/1e3:10.3f}, {final_pos[1]/1e3:10.3f}, {final_pos[2]/1e3:10.3f}]")
    print(f"Final Velocity (km/s): [{final_vel[0]/1e3:8.4f}, {final_vel[1]/1e3:8.4f}, {final_vel[2]/1e3:8.4f}]")
    print(f"Final Altitude:      {final_alt:.2f} km")
    print("=" * 70)

    return history


def main():
    parser = argparse.ArgumentParser(description="Orbital Mechanics SIL Simulator")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=None, help="Physics integration timestep in seconds")
    parser.add_argument("--inclination", type=float, default=None, help="Orbit inclination in degrees")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save output plot")
    parser.add_argument("--no-plot", action="store_true", help="Disable display of matplotlib plot window")
    args = parser.parse_args()

    cfg = SimConfig()
    if args.duration is not None:
        cfg.DURATION = args.duration
    if args.dt is not None:
        cfg.DT_PHYSICS = args.dt
    if args.inclination is not None:
        cfg.ORBIT_INCLINATION_DEG = args.inclination
        # Recalculate true initial state with new inclination
        cfg.TRUE_INITIAL_STATE = np.zeros(6)
        cfg.__post_init__()

    # Execute simulation
    history = run_simulation(cfg=cfg)

    # Post-run analysis & plotting
    if not args.no_plot:
        plot_orbit_and_residuals(
            history,
            r_earth=cfg.R_EARTH,
            save_path=args.save_plot,
            show=True,
        )
    elif args.save_plot:
        plot_orbit_and_residuals(
            history,
            r_earth=cfg.R_EARTH,
            save_path=args.save_plot,
            show=False,
        )


if __name__ == "__main__":
    main()

