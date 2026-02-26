import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from kalman_filter import KalmanFilter
from simulation import RobotSimulation2D

# --- CONFIG ---
TOTAL_TIME = 20.0   # Total simulation time (seconds)
IMU_DT = 0.05       # IMU timestep (seconds)
GPS_RATIO = 20      # How many IMU control steps per GPS measurement (e.g., 10 means 10 IMU steps = 1 GPS step)

ACCEL_X = 0.5   # Acceleration in X (m/s^2)
ACCEL_Y = 0.2   # Acceleration in Y (m/s^2)

IMU_NOISE_VAR = 0.005    # Variance of IMU acceleration noise (m/s^2)^2
GPS_NOISE_VAR = 0.5      # Variance of GPS position noise (m^2)
SIM_NOISE_VAR = 0.001    # Variance of noise added to the true state in the simulation (to make it more realistic)

GPS_CORRECTION_ENABLED = True # Can set to False to see how the filter drifts without GPS updates (pure dead reckoning)

SIM_SEED = None # Set to an integer for reproducibility, or None for random seed
# --- CONFIG ---

def run_demo():
    seed = SIM_SEED if SIM_SEED is not None else int(np.random.SeedSequence().entropy) % (2**32)
    np.random.seed(seed)
        
    print("Running Kalman Filter Demo with the following configuration:")
    print(f"TOTAL_TIME: {TOTAL_TIME} s")
    print(f"IMU_DT: {IMU_DT} s")
    print(f"GPS_RATIO: {GPS_RATIO} (1 GPS update every {GPS_RATIO} IMU steps)")
    print(f"ACCEL_X: {ACCEL_X} m/s^2")
    print(f"ACCEL_Y: {ACCEL_Y} m/s^2")
    print(f"IMU_NOISE_VAR: {IMU_NOISE_VAR} (m/s^2)^2")
    print(f"GPS_NOISE_VAR: {GPS_NOISE_VAR} m^2")
    print(f"SIM_NOISE_VAR: {SIM_NOISE_VAR} (m/s)^2")
    print(f"GPS_CORRECTION_ENABLED: {GPS_CORRECTION_ENABLED}")
    print(f"SIM_SEED: {seed}")
    print("-" * 50, "\n")

    num_steps = int(TOTAL_TIME / IMU_DT)

    plot_dir = os.path.join("2_kalman_filter", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Initialize filter with IMU timestep
    kf = KalmanFilter(IMU_DT, accel_var=IMU_NOISE_VAR, gps_var=GPS_NOISE_VAR)

    sim = RobotSimulation2D(IMU_DT)

    history = {"true": [], "meas": [], "kf": []}

    print(f"{'Step':<5} | {'True Pos (x,y)':<15} | {'KF Pos (x,y)':<15} | {'KF Vel (x,y)':<15}")
    print("-" * 65)

    u = np.array([[ACCEL_X], [ACCEL_Y]])

    for i in range(num_steps):

        # --- SIMULATION STEP ---
        true_state = sim.move(u, noise_std=np.sqrt(SIM_NOISE_VAR))

        # --- IMU PREDICT (runs every step) ---
        imu_reading = sim.get_imu_reading(u, noise_std=np.sqrt(IMU_NOISE_VAR))
        kf.predict(imu_reading)

        # --- GPS UPDATE (runs every GPS_RATIO IMU steps) ---
        if i % GPS_RATIO == 0:
            z = sim.get_gps_reading(noise_std=np.sqrt(GPS_NOISE_VAR))
            if GPS_CORRECTION_ENABLED:
                kf.correct_gps(z)
            meas = z.flatten()
        else:
            meas = [np.nan, np.nan]  # No GPS reading this step

        # Log
        history["true"].append(true_state[:2].flatten())
        history["meas"].append(meas)
        history["kf"].append(kf.x.flatten())

        if i % 10 == 0:
            print(
                f"{i:<5} | "
                f"{true_state[0,0]:>5.2f}, {true_state[1,0]:>5.2f} | "
                f"{kf.x[0,0]:>5.2f}, {kf.x[1,0]:>5.2f} | "
                f"{kf.x[2,0]:>5.2f}, {kf.x[3,0]:>5.2f}"
            )

    # --- Visualization ---
    true_hist = np.array(history["true"])
    meas_hist = np.array(history["meas"])
    kf_hist = np.array(history["kf"])

    plt.figure(figsize=(10, 8))

    plt.plot(true_hist[:,0], true_hist[:,1],
             'k-', label="True Path (Ground Truth)", linewidth=2)

    # Plot only valid GPS points
    valid = ~np.isnan(meas_hist[:,0])
    plt.scatter(meas_hist[valid,0], meas_hist[valid,1],
                c='red', alpha=0.5, s=20,
                label=f"Noisy GPS (GPS_NOISE_VAR={GPS_NOISE_VAR})")

    plt.plot(kf_hist[:,0], kf_hist[:,1],
             'g--', label="KF Estimate (Filtered)", linewidth=2)

    plt.title(
        f"4-State Kalman Filter: 2D Position & 2D Velocity\n"
        f"(IMU_NOISE_VAR={IMU_NOISE_VAR}, GPS_NOISE_VAR={GPS_NOISE_VAR})"
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plot_dir, f"kalman_filter_{timestamp}.png")
    plt.savefig(save_path)

    final_error = np.linalg.norm(true_state[:2] - kf.x[:2])
    print(f"\nFinal State Error: {final_error:.4f} m")
    print(f"Graph saved to: {save_path}")

    plt.close()


if __name__ == "__main__":
    run_demo()
