import numpy as np

class DroneSimulation3D:
    def __init__(self, dt, 
                 gyro_bias=np.array([0.02, -0.01, 0.005]), 
                 vibration_var=0.0005,
                 gyro_var=0.0001,
                 accel_var=0.005,
                 mag_var=0.01):
        self.dt = dt
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
        
        # True State: [phi (roll), theta (pitch), psi (yaw)]
        self.true_x = np.zeros(3)

    def _get_rotation_matrix(self, phi, theta, psi):
        """Body to Inertial frame rotation matrix (ZYX Euler)"""
        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        return R_z @ R_y @ R_x

    def step(self):
        """Simulate true dynamic motion with unmodeled vibration"""
        self.t += self.dt
        
        # Interesting flight dynamics
        p = 0.5 * np.cos(self.t)
        q = 0.3 * np.sin(self.t)
        r = 0.2
        
        # KINEMATIC DECOUPLING: Add vibration/turbulence the EKF can't see
        p += np.random.normal(0, self.vibration_std)
        q += np.random.normal(0, self.vibration_std)
        r += np.random.normal(0, self.vibration_std)
        
        phi, theta, psi = self.true_x
        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        
        self.true_x += np.array([phi_dot, theta_dot, psi_dot]) * self.dt
        self.true_x[2] = (self.true_x[2] + np.pi) % (2 * np.pi) - np.pi
        
        return self.true_x, np.array([p, q, r])

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
