import numpy as np

class MultiplicativeEKF:
    def __init__(self, dt, gyro_var=0.0001, accel_var=0.005, mag_var=0.01):
        self.dt = dt
        self.g = 9.81
        self.b_earth = np.array([1.0, 0.0, 0.0])
        
        # --- MEKF STATE ---
        # Nominal State: Unit Quaternion [qw, qx, qy, qz]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Error State Covariance (3x3 for small-angle error vector delta_theta)
        self.P = np.eye(3) * 0.1
        
        # --- TUNING ---
        self.Q = np.eye(3) * gyro_var
        self.R_acc = np.eye(3) * accel_var
        self.R_mag = np.eye(3) * mag_var

    @property
    def x(self):
        """Compatibility property: Returns Euler angles (rad) from the nominal quaternion."""
        qw, qx, qy, qz = self.q
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return np.array([roll, pitch, yaw])

    def _q_mult(self, q1, q2):
        """Hamiltonian Quaternion Multiplication"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def predict(self, u):
        """Phase A: Propagate Nominal Quaternion and 3D Error Covariance"""
        p, q, r = u
        
        # 1. Nominal State Propagation (Quaternion Kinematics)
        # Omega matrix for q_dot = 0.5 * Omega * q
        omega_quat = np.array([0, p, q, r])
        q_dot = 0.5 * self._q_mult(self.q, omega_quat)
        self.q = self.q + q_dot * self.dt
        self.q /= np.linalg.norm(self.q) # Re-normalize to prevent drift

        # 2. Error State Jacobian (F_t)
        # For MEKF, the error state jacobian is simpler (no tan/sec terms)
        # F_t = I - [omega_cross] * dt
        F_t = np.eye(3)
        F_t[0, 1] = r * self.dt
        F_t[0, 2] = -q * self.dt
        F_t[1, 0] = -r * self.dt
        F_t[1, 2] = p * self.dt
        F_t[2, 0] = q * self.dt
        F_t[2, 1] = -p * self.dt

        # 3. Covariance Propagation
        self.P = F_t @ self.P @ F_t.T + self.Q

    def _get_rot_matrix(self):
        """Converts nominal quaternion to Rotation Matrix (Inertial to Body)"""
        w, x, y, z = self.q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*x*y + 2*w*z,     2*x*z - 2*w*y],
            [    2*x*y - 2*w*z, 1 - 2*x**2 - 2*z**2,     2*y*z + 2*w*x],
            [    2*x*z + 2*w*y,     2*y*z - 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])

    def update_accel(self, z_acc):
        """Phase B: Correct using Accelerometer"""
        R = self._get_rot_matrix()
        
        # h(q) is gravity rotated into body frame
        gravity_inertial = np.array([0, 0, self.g])
        h_q = R @ gravity_inertial
        
        # H_t Jacobian: [h_q] cross product matrix (skew-symmetric)
        H_t = np.array([
            [0, -h_q[2], h_q[1]],
            [h_q[2], 0, -h_q[0]],
            [-h_q[1], h_q[0], 0]
        ])
        
        self._mekf_correction_update(z_acc, h_q, H_t, self.R_acc)

    def update_mag(self, z_mag):
        """Phase B: Correct using Magnetometer"""
        R = self._get_rot_matrix()
        
        # h(q) is earth mag field rotated into body frame
        h_q = R @ self.b_earth
        
        # H_t Jacobian: [h_q] cross product matrix (skew-symmetric)
        H_t = np.array([
            [0, -h_q[2], h_q[1]],
            [h_q[2], 0, -h_q[0]],
            [-h_q[1], h_q[0], 0]
        ])
        
        # Normalize measurement
        z_mag_norm = z_mag / np.linalg.norm(z_mag)
        self._mekf_correction_update(z_mag_norm, h_q, H_t, self.R_mag)

    def _mekf_correction_update(self, z, h_q, H_t, R_noise):
        """Multiplicative Injection Update"""
        # 1. Standard EKF math on the Error State
        y = z - h_q
        S = H_t @ self.P @ H_t.T + R_noise
        K = self.P @ H_t.T @ np.linalg.inv(S)
        
        # delta_theta is the 3D error vector
        delta_theta = K @ y
        
        # 2. Multiplicative Injection (Update nominal quaternion)
        # Convert 3D error vector to a small-angle delta quaternion
        # Using small angle approximation: sin(a) approx a, cos(a) approx 1
        alpha = delta_theta / 2.0
        dq = np.array([1.0, alpha[0], alpha[1], alpha[2]])
        dq /= np.linalg.norm(dq)
        
        self.q = self._q_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        
        # 3. Covariance Update
        self.P = (np.eye(3) - K @ H_t) @ self.P
