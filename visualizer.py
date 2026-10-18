"""
visualizer.py - 3D Orbital Trajectory & Estimation Residual Diagnostics

Renders the 3D orbital mesh and Earth sphere alongside 2D state estimation
error plots bounded by theoretical +-3sigma covariance envelopes.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import R_EARTH


def plot_3d_orbit(
    positions_km: np.ndarray,
    r_earth_km: float = R_EARTH / 1000.0,
    estimated_positions_km: Optional[np.ndarray] = None,
    measured_positions_km: Optional[np.ndarray] = None,
    title: str = "3D Orbital Trajectory (ECI Frame)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Renders a 3D Earth mesh and satellite trajectory in Earth-Centered Inertial (ECI) frame.

    Parameters:
        positions_km: Array of shape (N, 3) representing true satellite positions in km.
        r_earth_km: Earth radius in kilometers.
        estimated_positions_km: Optional array of shape (M, 3) representing EKF estimates.
        measured_positions_km: Optional array of shape (K, 3) representing raw sensor fixes.
        title: Title string for the 3D plot.
        ax: Existing 3D axis; if None, creates a new figure and axis.

    Returns:
        plt.Axes: The 3D plot axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

    # 1. Earth sphere mesh (parametric surface)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x_earth = r_earth_km * np.outer(np.cos(u), np.sin(v))
    y_earth = r_earth_km * np.outer(np.sin(u), np.sin(v))
    z_earth = r_earth_km * np.outer(np.ones_like(u), np.cos(v))

    # Plot Earth surface (semi-transparent royal blue)
    ax.plot_surface(
        x_earth,
        y_earth,
        z_earth,
        color="#1f77b4",
        alpha=0.30,
        edgecolor="#0a2f5c",
        linewidth=0.2,
        rstride=2,
        cstride=2,
    )

    # Earth equatorial reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        r_earth_km * np.cos(theta),
        r_earth_km * np.sin(theta),
        np.zeros_like(theta),
        color="#0a2f5c",
        linestyle="--",
        linewidth=0.8,
        label="Equator",
    )

    # 2. Downsample for smooth interactive rendering if trajectory is large
    n_pts = len(positions_km)
    step = max(1, n_pts // 3000)
    pos_sub = positions_km[::step]

    # Plot ground truth trajectory
    ax.plot(
        pos_sub[:, 0],
        pos_sub[:, 1],
        pos_sub[:, 2],
        color="crimson",
        linewidth=1.8,
        label="Ground Truth Orbit",
    )

    # Start and End markers
    ax.scatter(
        positions_km[0, 0],
        positions_km[0, 1],
        positions_km[0, 2],
        color="green",
        s=60,
        edgecolor="black",
        zorder=5,
        label="Start",
    )
    ax.scatter(
        positions_km[-1, 0],
        positions_km[-1, 1],
        positions_km[-1, 2],
        color="darkred",
        marker="s",
        s=60,
        edgecolor="black",
        zorder=5,
        label="End / Current",
    )

    # Optional: Plot raw sensor fixes if available
    if measured_positions_km is not None and len(measured_positions_km) > 0:
        meas_step = max(1, len(measured_positions_km) // 500)
        m_sub = measured_positions_km[::meas_step]
        ax.scatter(
            m_sub[:, 0],
            m_sub[:, 1],
            m_sub[:, 2],
            color="goldenrod",
            s=12,
            alpha=0.5,
            label="Sensor Fixes",
        )

    # Optional: Plot filter state estimate
    if estimated_positions_km is not None and len(estimated_positions_km) > 0:
        est_step = max(1, len(estimated_positions_km) // 3000)
        est_sub = estimated_positions_km[::est_step]
        ax.plot(
            est_sub[:, 0],
            est_sub[:, 1],
            est_sub[:, 2],
            color="dodgerblue",
            linestyle="--",
            linewidth=1.4,
            label="EKF Estimate",
        )

    # 3. Lock aspect ratio so Earth remains spherical
    ax.set_box_aspect([1, 1, 1])

    # 4. Compute bounding envelope
    max_val = max(np.max(np.abs(positions_km)), r_earth_km * 1.15)
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    ax.set_xlabel("ECI X (km)", fontsize=10)
    ax.set_ylabel("ECI Y (km)", fontsize=10)
    ax.set_zlabel("ECI Z (km)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=8)

    return ax


def plot_residuals(
    time_array: np.ndarray,
    truth_array: np.ndarray,
    est_array: np.ndarray,
    cov_array: Optional[np.ndarray] = None,
    axs: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Plots estimation residuals (Error = Estimate - Truth) along with theoretical +-3sigma bounds.

    Parameters:
        time_array: Array of simulation timestamps (seconds).
        truth_array: Array of true state vectors (N, 6).
        est_array: Array of estimated state vectors (N, 6).
        cov_array: Optional array of covariance diagonals (N, 6) [var_x, var_y, var_z, var_vx, ...].
        axs: Optional pair of Axes [ax_pos, ax_vel].

    Returns:
        np.ndarray: The array of axes used for plotting.
    """
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_pos, ax_vel = axs[0], axs[1]

    # Downsample if series is dense
    n_pts = len(time_array)
    step = max(1, n_pts // 2000)
    t = time_array[::step]
    truth = truth_array[::step]
    est = est_array[::step]

    # Compute errors (Estimate - Truth)
    pos_err = est[:, :3] - truth[:, :3]     # meters
    vel_err = est[:, 3:6] - truth[:, 3:6]   # m/s

    sigma = np.sqrt(cov_array[::step]) if cov_array is not None and len(cov_array) > 0 else None

    # Colors for X, Y, Z axes
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["X", "Y", "Z"]

    # Position residuals plot
    for i in range(3):
        ax_pos.plot(t, pos_err[:, i], color=colors[i], label=f"Error {labels[i]}", linewidth=1.2)
        if sigma is not None:
            bound = 3.0 * sigma[:, i]
            ax_pos.fill_between(
                t, -bound, bound, color=colors[i], alpha=0.15, label=f"+-3sigma {labels[i]}" if i == 0 else None
            )
            ax_pos.plot(t, bound, color=colors[i], linestyle=":", linewidth=0.8)
            ax_pos.plot(t, -bound, color=colors[i], linestyle=":", linewidth=0.8)

    ax_pos.set_ylabel("Position Error (m)", fontsize=10)
    ax_pos.set_title("Position Estimation Error with +-3sigma Filter Bounds", fontsize=11)
    ax_pos.grid(True, linestyle="--", alpha=0.6)
    ax_pos.legend(loc="upper right", fontsize=8, ncol=2)

    # Velocity residuals plot
    for i in range(3):
        ax_vel.plot(t, vel_err[:, i], color=colors[i], label=f"Error V{labels[i].lower()}", linewidth=1.2)
        if sigma is not None:
            bound = 3.0 * sigma[:, 3 + i]
            ax_vel.fill_between(
                t, -bound, bound, color=colors[i], alpha=0.15, label=f"+-3sigma V{labels[i].lower()}" if i == 0 else None
            )
            ax_vel.plot(t, bound, color=colors[i], linestyle=":", linewidth=0.8)
            ax_vel.plot(t, -bound, color=colors[i], linestyle=":", linewidth=0.8)

    ax_vel.set_xlabel("Mission Elapsed Time (s)", fontsize=10)
    ax_vel.set_ylabel("Velocity Error (m/s)", fontsize=10)
    ax_vel.set_title("Velocity Estimation Error with +-3sigma Filter Bounds", fontsize=11)
    ax_vel.grid(True, linestyle="--", alpha=0.6)
    ax_vel.legend(loc="upper right", fontsize=8, ncol=2)

    return axs


def plot_diagnostics(
    time_array: np.ndarray,
    truth_array: np.ndarray,
    r_earth_m: float = R_EARTH,
    axs: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Renders telemetry diagnostics (altitude and orbital speed vs time)
    when estimator/sensors telemetry is not yet active.
    """
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_alt, ax_spd = axs[0], axs[1]

    n_pts = len(time_array)
    step = max(1, n_pts // 2000)
    t = time_array[::step]
    truth = truth_array[::step]

    # Altitude: ||r|| - R_earth in km
    r_mag = np.linalg.norm(truth[:, :3], axis=1)
    altitude_km = (r_mag - r_earth_m) / 1000.0

    # Orbital speed: ||v|| in km/s
    speed_km_s = np.linalg.norm(truth[:, 3:6], axis=1) / 1000.0

    ax_alt.plot(t, altitude_km, color="royalblue", linewidth=1.5)
    ax_alt.set_ylabel("Altitude (km)", fontsize=10)
    ax_alt.set_title("Orbital Altitude Profile (J2 Perturbed)", fontsize=11)
    ax_alt.grid(True, linestyle="--", alpha=0.6)

    ax_spd.plot(t, speed_km_s, color="crimson", linewidth=1.5)
    ax_spd.set_xlabel("Mission Elapsed Time (s)", fontsize=10)
    ax_spd.set_ylabel("Orbital Speed (km/s)", fontsize=10)
    ax_spd.set_title("Orbital Speed Profile", fontsize=11)
    ax_spd.grid(True, linestyle="--", alpha=0.6)

    return axs


def plot_orbit_and_residuals(
    history: Dict[str, Any],
    r_earth: float = R_EARTH,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Unified visualization dashboard per specs.md:
    Displays 3D orbit trajectory with Earth mesh alongside 2D estimation residuals
    bounded by theoretical +-3sigma filter envelopes.

    Parameters:
        history: Telemetry dictionary containing:
            - 'time': List or array of timestamps (s)
            - 'truth': List or array of 6D truth state vectors [x, y, z, vx, vy, vz]
            - 'meas': Optional list of tuples (t, z) or array of raw sensor fixes
            - 'est': Optional list or array of 6D estimated state vectors
            - 'cov': Optional list or array of covariance diagonal variances
        r_earth: Earth equatorial radius (meters).
        save_path: Optional file path to save the generated figure (e.g. .png).
        show: Whether to call plt.show().

    Returns:
        plt.Figure: The rendered matplotlib Figure.
    """
    time_array = np.asarray(history.get("time", []))
    truth_array = np.asarray(history.get("truth", []))

    if len(truth_array) == 0:
        raise ValueError("Telemetry history['truth'] is empty. Nothing to plot.")

    # Convert positions to km for cleaner 3D coordinates
    positions_km = truth_array[:, :3] / 1000.0
    r_earth_km = r_earth / 1000.0

    # Extract optional flight software estimation telemetry
    has_est = False
    est_positions_km = None
    est_array = None
    if "est" in history and len(history["est"]) > 0:
        est_array = np.asarray(history["est"])
        if len(est_array) == len(truth_array) and not np.all(est_array == 0):
            est_positions_km = est_array[:, :3] / 1000.0
            has_est = True

    cov_array = np.asarray(history["cov"]) if ("cov" in history and len(history["cov"]) > 0) else None

    # Extract optional sensor measurements
    meas_positions_km = None
    if "meas" in history and len(history["meas"]) > 0:
        meas_raw = history["meas"]
        if isinstance(meas_raw[0], (tuple, list)) and len(meas_raw[0]) == 2:
            # Format: [(t, z_k), ...]
            meas_vecs = np.array([m[1][:3] for m in meas_raw])
        else:
            meas_vecs = np.array([m[:3] for m in meas_raw])
        meas_positions_km = meas_vecs / 1000.0

    # Setup layout
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0])

    # 3D Orbit Axis on left side
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    plot_3d_orbit(
        positions_km=positions_km,
        r_earth_km=r_earth_km,
        estimated_positions_km=est_positions_km,
        measured_positions_km=meas_positions_km,
        title="3D Orbital Motion (ECI Reference Frame)",
        ax=ax_3d,
    )

    # 2D Diagnostics on right side
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)

    if has_est:
        plot_residuals(
            time_array=time_array,
            truth_array=truth_array,
            est_array=est_array,
            cov_array=cov_array,
            axs=[ax_top, ax_bot],
        )
    else:
        plot_diagnostics(
            time_array=time_array,
            truth_array=truth_array,
            r_earth_m=r_earth,
            axs=[ax_top, ax_bot],
        )
        ax_top.text(
            0.5,
            0.90,
            "Estimator & Sensors pending implementation\nDisplaying Truth Dynamics Telemetry",
            transform=ax_top.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba"),
        )

    plt.suptitle("Orbital Simulator Telemetry & SIL Mission Dashboard", fontsize=14, weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualizer] Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig

