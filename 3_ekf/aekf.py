import numpy as np

class AdditiveEKF:
    def __init__(self, dt, gyro_var=0.0001, accel_var=0.005, mag_var=0.01):
        self.dt = dt
        self.g = 9.81
        self.b_earth = np.array([1.0, 0.0, 0.0])
        
        # State: [roll, pitch, yaw]
        self.x = np.zeros(3)
        self.P = np.eye(3) * 1.0  # Initial uncertainty
        
        # Tuning Matrices
        self.Q = np.eye(3) * gyro_var       # Process noise (trust in gyro)
        self.R_acc = np.eye(3) * accel_var  # Accel noise
        self.R_mag = np.eye(3) * mag_var    # Mag noise

    def _f_state_transition(self, x, u):
        """Nonlinear state transition function f(x, u)"""
        phi, theta, psi = x
        p, q, r = u
        
        phi_rate = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_rate = q * np.cos(phi) - r * np.sin(phi)
        psi_rate = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        
        rates = np.array([phi_rate, theta_rate, psi_rate])
        return x + rates * self.dt # First order discretization

    def predict(self, u):
        """Phase A: Predict using Gyro"""
        # Get current state for use in Jacobian (x_{t-1})
        phi, theta, psi = self.x
        
        # 1. State Prediction
        self.x = self._f_state_transition(self.x, u)
        
        # Wrap all angles to [-pi, pi]
        self.x[0] = (self.x[0] + np.pi) % (2 * np.pi) - np.pi
        self.x[1] = (self.x[1] + np.pi) % (2 * np.pi) - np.pi
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Jacobian F_t (Analytical approximation: I + df/dx * dt)
        p, q, r = u

        # Precompute trigonometric terms for Jacobian
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        ttheta = np.tan(theta)
        ctheta = np.cos(theta)
        sectheta = 1.0 / ctheta
        sec2theta = sectheta**2

        # Compute the Jacobian of the rates with respect to the state variables (g function in the notes)
        rates_jacobian = np.zeros((3, 3))

        # ---- Row 1 (phi) ----
        rates_jacobian[0, 0] += (q * cphi * ttheta - r * sphi * ttheta)
        rates_jacobian[0, 1] += ((q * sphi + r * cphi) * sec2theta)
        rates_jacobian[0, 2] += 0

        # ---- Row 2 (theta) ----
        rates_jacobian[1, 0] += (-q * sphi - r * cphi)
        rates_jacobian[1, 1] += 0
        rates_jacobian[1, 2] += 0

        # ---- Row 3 (psi) ----
        rates_jacobian[2, 0] += ((q * cphi - r * sphi) * sectheta)
        rates_jacobian[2, 1] += ((q * sphi + r * cphi) * sectheta * ttheta)
        rates_jacobian[2, 2] += 0
        
        # Final F_t Jacobian
        F_t = np.eye(3) + (rates_jacobian * self.dt) # Since we are using first order discretization, F_t = I + dg/dx * dt
        
        # 3. Covariance Prediction
        self.P = F_t @ self.P @ F_t.T + self.Q

    def update_accel(self, z_acc):
        """Phase B: Correct using Accelerometer"""
        phi, theta, psi = self.x
        
        # h_accel(x)
        h_x = np.array([
            -self.g * np.sin(theta),
            self.g * np.cos(theta) * np.sin(phi),
            self.g * np.cos(theta) * np.cos(phi)
        ])
        
        # Jacobian H_accel (Analytical from notes)
        H_t = np.array([
            [0, -self.g * np.cos(theta), 0],
            [self.g * np.cos(theta) * np.cos(phi), -self.g * np.sin(theta) * np.sin(phi), 0],
            [-self.g * np.cos(theta) * np.sin(phi), -self.g * np.sin(theta) * np.cos(phi), 0]
        ])
        
        # EKF Correction Update
        self._ekf_correction_update(z_acc, h_x, H_t, self.R_acc)

    def update_mag(self, z_mag):
        """Phase B: Correct using Magnetometer"""
        # Normalize magnetometer measurement
        z_mag = z_mag / np.linalg.norm(z_mag)
        
        # h_mag(x)
        R_x = np.array([[1, 0, 0], [0, np.cos(self.x[0]), -np.sin(self.x[0])], [0, np.sin(self.x[0]), np.cos(self.x[0])]])
        R_y = np.array([[np.cos(self.x[1]), 0, np.sin(self.x[1])], [0, 1, 0], [-np.sin(self.x[1]), 0, np.cos(self.x[1])]])
        R_z = np.array([[np.cos(self.x[2]), -np.sin(self.x[2]), 0], [np.sin(self.x[2]), np.cos(self.x[2]), 0], [0, 0, 1]])
        R_inertial_to_body = (R_z @ R_y @ R_x).T
        
        h_x = R_inertial_to_body @ self.b_earth
        
        # Jacobian H_mag (Numerical Finite Difference approach)
        H_t = np.zeros((3, 3))
        epsilon = 1e-5
        for i in range(3):
            x_pert = self.x.copy()
            x_pert[i] += epsilon
            
            R_x_p = np.array([[1, 0, 0], [0, np.cos(x_pert[0]), -np.sin(x_pert[0])], [0, np.sin(x_pert[0]), np.cos(x_pert[0])]])
            R_y_p = np.array([[np.cos(x_pert[1]), 0, np.sin(x_pert[1])], [0, 1, 0], [-np.sin(x_pert[1]), 0, np.cos(x_pert[1])]])
            R_z_p = np.array([[np.cos(x_pert[2]), -np.sin(x_pert[2]), 0], [np.sin(x_pert[2]), np.cos(x_pert[2]), 0], [0, 0, 1]])
            
            h_x_pert = (R_z_p @ R_y_p @ R_x_p).T @ self.b_earth
            H_t[:, i] = (h_x_pert - h_x) / epsilon

        # EKF Correction Update
        self._ekf_correction_update(z_mag, h_x, H_t, self.R_mag)
        
    def _ekf_correction_update(self, z, h_x, H_t, R):
        """Generic EKF correction step"""
        y = z - h_x
        S = H_t @ self.P @ H_t.T + R
        K = self.P @ H_t.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H_t) @ self.P
