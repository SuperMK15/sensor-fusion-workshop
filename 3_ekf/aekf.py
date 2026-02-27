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

    def f_kinematics(self, x, u):
        """Nonlinear state transition function f(x, u)"""
        phi, theta, psi = x
        p, q, r = u
        
        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        
        return np.array([phi_dot, theta_dot, psi_dot])

    def predict(self, u):
        """Phase A: Predict using Gyro"""
        # 1. State Prediction
        rates = self.f_kinematics(self.x, u)
        self.x = self.x + rates * self.dt
        
        # Wrap yaw
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Jacobian F_t (Analytical approximation: I + df/dx * dt)
        phi, theta, psi = self.x
        p, q, r = u
        
        # Partial derivatives of rates w.r.t phi, theta, psi
        F_jac = np.eye(3)
        # (Simplified Jacobian calculation for brevity - a real flight controller uses full derivatives)
        F_jac[0, 0] += (q * np.cos(phi) * np.tan(theta) - r * np.sin(phi) * np.tan(theta)) * self.dt
        F_jac[0, 1] += (q * np.sin(phi) / np.cos(theta)**2 + r * np.cos(phi) / np.cos(theta)**2) * self.dt
        F_jac[1, 0] += (-q * np.sin(phi) - r * np.cos(phi)) * self.dt
        F_jac[2, 0] += (q * np.cos(phi) / np.cos(theta) - r * np.sin(phi) / np.cos(theta)) * self.dt
        F_jac[2, 1] += (q * np.sin(phi) * np.tan(theta) / np.cos(theta) + r * np.cos(phi) * np.tan(theta) / np.cos(theta)) * self.dt

        # 3. Covariance Prediction
        self.P = F_jac @ self.P @ F_jac.T + self.Q

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
