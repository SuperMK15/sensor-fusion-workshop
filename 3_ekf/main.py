import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from aekf import AdditiveEKF
from simulation import DroneSimulation3D

# --- CONFIG ---
TOTAL_TIME = 50.0   # Total simulation time (seconds)
GYRO_DT = 0.01      # Gyro / Prediction timestep (seconds) - 100Hz

# Sensor Rates
ACCEL_RATIO = 1 # Default: 1 Accel update every 1 Gyro step (Standard 6-DOF IMU behavior)
MAG_RATIO = 5   # Default: 1 Mag update every 5 Gyro steps (20Hz magnetometer)

# Gyro Bias (constant drift added to gyro measurements, unknown to EKF)
GYRO_BIAS_X = 0.02  # (rad/s)
GYRO_BIAS_Y = -0.01 # (rad/s)
GYRO_BIAS_Z = 0.005 # (rad/s)

# Noise Variances
GYRO_NOISE_VAR = 0.0001  # Variance of Gyro noise (rad/s)^2 - acts as Q matrix
ACCEL_NOISE_VAR = 0.005  # Variance of Accel noise (m/s^2)^2 - acts as R_acc matrix
MAG_NOISE_VAR = 0.01     # Variance of Mag noise (normalized)^2 - acts as R_mag matrix
SIM_NOISE_VAR = 0.0005   # Variance of physical disturbance added to true state (for realism)

# Toggles for Fusion
ACCEL_CORRECTION_ENABLED = True
MAG_CORRECTION_ENABLED = True

SIM_SEED = None # Set to an int for reproducibility, or None for random
# --- CONFIG ---

def run_aekf_demo():
    # Setup Random Seed
    seed = SIM_SEED if SIM_SEED is not None else int(np.random.SeedSequence().entropy) % (2**32)
    np.random.seed(seed)
        
    print("Running AEKF Demo with the following configuration:")
    print(f"TOTAL_TIME: {TOTAL_TIME} s")
    print(f"GYRO_DT: {GYRO_DT} s")
    print(f"ACCEL_RATIO: {ACCEL_RATIO} (1 Accel update every {ACCEL_RATIO} Gyro steps)")
    print(f"MAG_RATIO: {MAG_RATIO} (1 Mag update every {MAG_RATIO} Gyro steps)")
    print(f"GYRO_NOISE_VAR: {GYRO_NOISE_VAR}")
    print(f"ACCEL_NOISE_VAR: {ACCEL_NOISE_VAR}")
    print(f"MAG_NOISE_VAR: {MAG_NOISE_VAR}")
    print(f"ACCEL_CORRECTION_ENABLED: {ACCEL_CORRECTION_ENABLED}")
    print(f"MAG_CORRECTION_ENABLED: {MAG_CORRECTION_ENABLED}")
    print(f"SIM_SEED: {seed}")
    print("-" * 65, "\n")

    num_steps = int(TOTAL_TIME / GYRO_DT)

    # --- PLOT DIRECTORY SETUP ---
    # Get the directory where aekf_main.py is physically located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the 'plots' folder relative to this script's directory
    plot_dir = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # -----------------------------

    # Initialize simulation and filter
    sim = DroneSimulation3D(GYRO_DT,
                            gyro_bias=np.array([GYRO_BIAS_X, GYRO_BIAS_Y, GYRO_BIAS_Z]),
                            vibration_var=SIM_NOISE_VAR,
                            gyro_var=GYRO_NOISE_VAR,
                            accel_var=ACCEL_NOISE_VAR,
                            mag_var=MAG_NOISE_VAR)
    ekf = AdditiveEKF(GYRO_DT,
                      gyro_var=GYRO_NOISE_VAR,
                      accel_var=ACCEL_NOISE_VAR,
                      mag_var=MAG_NOISE_VAR)

    # Storage for plotting
    history = {"t": [], "true_euler": [], "ekf_euler": [], "z_acc": [], "z_mag": [], "u_gyro": []}

    print(f"{'Step':<5} | {'True [Roll, Pitch, Yaw]°':<25} | {'EKF [Roll, Pitch, Yaw]°':<25}")
    print("-" * 65)

    for i in range(num_steps):
        # --- SIMULATION STEP ---
        true_euler, true_rates = sim.step()

        # --- PREDICT (runs every step) ---
        u_gyro = sim.get_gyro(true_rates)
        ekf.predict(u_gyro)

        # --- ACCELEROMETER UPDATE ---
        if i % ACCEL_RATIO == 0:
            z_acc = sim.get_accel()
            if ACCEL_CORRECTION_ENABLED:
                ekf.update_accel(z_acc)
            acc_log = z_acc.copy()
        else:
            acc_log = [np.nan, np.nan, np.nan]

        # --- MAGNETOMETER UPDATE ---
        if i % MAG_RATIO == 0:
            z_mag = sim.get_mag()
            if MAG_CORRECTION_ENABLED:
                ekf.update_mag(z_mag)
            mag_log = z_mag.copy()
        else:
            mag_log = [np.nan, np.nan, np.nan]

        # --- LOGGING ---
        history["t"].append(sim.t)
        history["true_euler"].append(np.degrees(true_euler.copy()))
        history["ekf_euler"].append(np.degrees(ekf.x.copy()))
        history["u_gyro"].append(u_gyro.copy())
        history["z_acc"].append(acc_log)
        history["z_mag"].append(mag_log)

        # Adjusted print frequency for 100Hz
        if i % 100 == 0:
            tr = np.degrees(true_euler)
            est = np.degrees(ekf.x)
            print(
                f"{i:<5} | "
                f"{tr[0]:>6.1f}, {tr[1]:>6.1f}, {tr[2]:>6.1f} | "
                f"{est[0]:>6.1f}, {est[1]:>6.1f}, {est[2]:>6.1f}"
            )

    # --- VISUALIZATION ---
    t = np.array(history["t"])
    true_e = np.array(history["true_euler"])
    est_e = np.array(history["ekf_euler"])
    u_gyro = np.array(history["u_gyro"])
    acc = np.array(history["z_acc"])
    mag = np.array(history["z_mag"])
    
    # Calculate absolute error in degrees
    err = np.abs(true_e - est_e)
    err[:, 2] = (err[:, 2] + 180) % 360 - 180
    err = np.abs(err)

    fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    
    # Plot 1: Absolute State Error
    axs[0].plot(t, err[:, 0], 'r', label="Roll Err")
    axs[0].plot(t, err[:, 1], 'g', label="Pitch Err")
    axs[0].plot(t, err[:, 2], 'b', label="Yaw Err")
    axs[0].set_title("Absolute State Error")
    axs[0].set_ylabel("Error (Degrees)")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Attitude (Lines)
    axs[1].plot(t, true_e[:, 0], 'r-', label="True Roll", linewidth=2)
    axs[1].plot(t, est_e[:, 0], 'r--', label="EKF Roll", linewidth=2)
    axs[1].plot(t, true_e[:, 1], 'g-', label="True Pitch", linewidth=2)
    axs[1].plot(t, est_e[:, 1], 'g--', label="EKF Pitch", linewidth=2)
    axs[1].plot(t, true_e[:, 2], 'b-', label="True Yaw", linewidth=2)
    axs[1].plot(t, est_e[:, 2], 'b--', label="EKF Yaw", linewidth=2)
    axs[1].set_title("AEKF 3D Attitude Estimation")
    axs[1].set_ylabel("Angle (Degrees)")
    axs[1].legend(loc='upper right', ncol=3)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: Gyro Inputs (Lines)
    axs[2].plot(t, u_gyro[:, 0], 'r', alpha=0.7, label="Gyro P")
    axs[2].plot(t, u_gyro[:, 1], 'g', alpha=0.7, label="Gyro Q")
    axs[2].plot(t, u_gyro[:, 2], 'b', alpha=0.7, label="Gyro R")
    axs[2].set_title("Raw Gyroscope Inputs (with Bias/Noise)")
    axs[2].set_ylabel("Rad/s")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle=':', alpha=0.6)

    # Plot 4: Accelerometer Data (Scatter for discrete updates)
    valid_acc = ~np.isnan(acc[:, 0])
    axs[3].scatter(t[valid_acc], acc[valid_acc, 0], c='r', alpha=0.2, s=5, label="Accel X")
    axs[3].scatter(t[valid_acc], acc[valid_acc, 1], c='g', alpha=0.2, s=5, label="Accel Y")
    axs[3].scatter(t[valid_acc], acc[valid_acc, 2], c='b', alpha=0.2, s=5, label="Accel Z")
    axs[3].set_title(f"Raw Accelerometer (Ratio 1:{ACCEL_RATIO})")
    axs[3].set_ylabel("Acceleration (m/s²)")
    axs[3].legend(loc='upper right', ncol=3)
    axs[3].grid(True, linestyle=':', alpha=0.6)

    # Plot 5: Magnetometer Data (Scatter for discrete updates)
    valid_mag = ~np.isnan(mag[:, 0])
    axs[4].scatter(t[valid_mag], mag[valid_mag, 0], c='r', alpha=0.4, s=15, label="Mag X")
    axs[4].scatter(t[valid_mag], mag[valid_mag, 1], c='g', alpha=0.4, s=15, label="Mag Y")
    axs[4].scatter(t[valid_mag], mag[valid_mag, 2], c='b', alpha=0.4, s=15, label="Mag Z")
    axs[4].set_title(f"Raw Magnetometer (Ratio 1:{MAG_RATIO})")
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Normalized Field")
    axs[4].legend(loc='upper right', ncol=3)
    axs[4].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plot_dir, f"aekf_results_{timestamp}.png")
    plt.savefig(save_path)

    # Calculate final error in degrees
    final_error = np.abs(true_e[-1] - est_e[-1])
    # Handle yaw wrap-around for error calculation
    final_error[2] = (final_error[2] + 180) % 360 - 180
    mae = np.mean(np.abs(final_error))
    
    print(f"\nFinal State Error (Mean Absolute): {mae:.2f}°")
    print(f"Graph saved to: {save_path}")

    plt.close()

if __name__ == "__main__":
    run_aekf_demo()
