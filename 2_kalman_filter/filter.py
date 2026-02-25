import numpy as np

class KalmanFilter:
    def __init__(self, dt, accel_var, gps_var):
        self.dt = dt
        
        # State Vector [px, py, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Initial Uncertainty P (High values mean we don't know initial state)
        self.P = np.eye(4) * 1.0
        
        # State Transition Matrix F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Control Matrix B (Input: [ax, ay])
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        # Observation Matrix H (GPS only sees px, py)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 1. Initialize R (Measurement Noise)
        # Based on GPS variance
        self.R = np.eye(2) * gps_var
        
        # 2. Initialize Q (Process Noise)
        # Using the Piecewise White Noise model derived from IMU acceleration noise
        self.Q = np.array([
            [0.25*dt**4, 0,          0.5*dt**3,  0         ],
            [0,          0.25*dt**4,  0,         0.5*dt**3 ],
            [0.5*dt**3,  0,           dt**2,     0         ],
            [0,          0.5*dt**3,  0,          dt**2     ]
        ]) * accel_var

    def predict(self, u):
        """Phase A: Prediction (IMU Step)"""
        self.x = self.F @ self.x + self.B @ u # State prediction using control input (acceleration)
        self.P = self.F @ self.P @ self.F.T + self.Q # Uncertainty prediction
        return self.x, self.P

    def correct_gps(self, z):
        """Phase B: Correction (GPS Step)"""
        S = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        y = z - self.H @ self.x # Measurement residual (innovation)
        self.x = self.x + K @ y # State update with measurement
        
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P # Uncertainty update
        return self.x, self.P
