import numpy as np

class RobotSimulation2D:
    def __init__(self, dt):
        self.dt = dt
        self.true_x = np.zeros((4, 1)) # [px, py, vx, vy]
        
    def move(self, accel_cmd, noise_std=0.05):
        """Simulate physical movement with noise"""
        # Simple kinematics
        ax, ay = accel_cmd[0,0], accel_cmd[1,0]
        
        # Use current velocity for position update before modifying it
        vx, vy = self.true_x[2,0], self.true_x[3,0]
        
        # Update Position
        self.true_x[0,0] += vx * self.dt + 0.5 * ax * self.dt**2
        self.true_x[1,0] += vy * self.dt + 0.5 * ay * self.dt**2
        
        # Update Velocity
        self.true_x[2,0] += ax * self.dt + np.random.normal(0, noise_std)
        self.true_x[3,0] += ay * self.dt + np.random.normal(0, noise_std)
        
        return self.true_x
    
    def get_imu_reading(self, true_accel, noise_std=0.1):
        """Simulates an accelerometer reading with Gaussian noise"""
        ax_noisy = true_accel[0,0] + np.random.normal(0, noise_std)
        ay_noisy = true_accel[1,0] + np.random.normal(0, noise_std)
        return np.array([[ax_noisy], [ay_noisy]])

    def get_gps_reading(self, noise_std=0.5):
        """Returns noisy 2D position"""
        z = self.true_x[:2] + np.random.normal(0, noise_std, (2, 1))
        return z
