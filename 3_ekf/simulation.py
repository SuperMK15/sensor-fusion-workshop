import numpy as np
from enum import Enum, auto

class SimMode(Enum):
    OSCILLATE = auto()    # Standard flight dynamics
    CONSTANT_ROT = auto() # Constant rotation rates (can trigger Gimbal Lock!)

class DroneSimulation3D:
    def __init__(self, dt, 
                 mode=SimMode.OSCILLATE,
                 const_rates=np.array([0.1, 0.5, 0.1]),
                 osc_amplitudes=np.array([0.5, 0.3, 0.2]),
                 gyro_bias=np.array([0.02, -0.01, 0.005]), 
                 vibration_var=0.0005,
                 gyro_var=0.0001,
                 accel_var=0.005,
                 mag_var=0.01):
        self.dt = dt
        self.mode = mode
        self.const_rates = const_rates
        self.osc_amplitudes = osc_amplitudes
        self.t = 0.0
        self.g = 9.81
        self.b_earth = np.array([1.0, 0.0, 0.0])
        
        # --- DECOUPLING PARAMETERS ---
        self.gyro_bias = gyro_bias # Constant "drift" added to gyro
        self.vibration_std = np.sqrt(vibration_var) # Unmodeled vibration added to true state
        
        # Sensor noise levels
        self.gyro_std = np.sqrt(gyro_var)
        self.accel_std = np.sqrt(accel_var)
        self.mag_std = np.sqrt(mag_var)
        
        # True State: Use Quaternion internally to avoid Gimbal Lock
        self.true_x = np.zeros(3) # [Roll, Pitch, Yaw]
        self.true_q = np.array([1.0, 0.0, 0.0, 0.0])

    def _get_rotation_matrix(self, phi, theta, psi):
        """Body to Inertial frame rotation matrix (ZYX Euler)"""
        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        return R_z @ R_y @ R_x

    def step(self):
        """Simulate true dynamic motion with unmodeled vibration"""
        self.t += self.dt
        
        if self.mode == SimMode.CONSTANT_ROT:
            p, q, r = self.const_rates
        else:
            p = self.osc_amplitudes[0] * np.cos(self.t)
            q = self.osc_amplitudes[1] * np.sin(self.t)
            r = self.osc_amplitudes[2]
        
        p_noisy = p + np.random.normal(0, self.vibration_std)
        q_noisy = q + np.random.normal(0, self.vibration_std)
        r_noisy = r + np.random.normal(0, self.vibration_std)
        
        # Update True Quaternion (Singularity-Free Integration)
        w, x, y, z = self.true_q
        q_dot = 0.5 * np.array([
            -x*p_noisy - y*q_noisy - z*r_noisy,
             w*p_noisy + y*r_noisy - z*q_noisy,
             w*q_noisy - x*r_noisy + z*p_noisy,
             w*r_noisy + x*q_noisy - y*p_noisy
        ])
        
        self.true_q += q_dot * self.dt
        self.true_q /= np.linalg.norm(self.true_q)
        
        # Extract Euler Angles for sensor methods (Normalized to -pi, pi)
        qw, qx, qy, qz = self.true_q
        self.true_x[0] = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        self.true_x[1] = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        self.true_x[2] = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        return self.true_x.copy(), np.array([p, q, r])

    def get_gyro(self, true_rates):
        """Gyro returns rates with persistent BIAS + Noise"""
        return true_rates + self.gyro_bias + np.random.normal(0, self.gyro_std, 3)

    def get_accel(self):
        """Accelerometer measures gravity projection in body frame"""
        phi, theta, psi = self.true_x
        a_x = -self.g * np.sin(theta)
        a_y = self.g * np.cos(theta) * np.sin(phi)
        a_z = self.g * np.cos(theta) * np.cos(phi)
        return np.array([a_x, a_y, a_z]) + np.random.normal(0, self.accel_std, 3)

    def get_mag(self):
        """Magnetometer measures Earth's magnetic field in body frame"""
        phi, theta, psi = self.true_x
        R_body_to_inertial = self._get_rotation_matrix(phi, theta, psi)
        R_inertial_to_body = R_body_to_inertial.T
        m_body = R_inertial_to_body @ self.b_earth
        return m_body + np.random.normal(0, self.mag_std, 3)
