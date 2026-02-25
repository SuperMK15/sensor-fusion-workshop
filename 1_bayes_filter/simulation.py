import numpy as np

class RobotSimulation:
    def __init__(self, actual_start_pos, wall_pos):
        self.actual_x = actual_start_pos
        self.wall_pos = wall_pos

    def move(self, move_dist, move_noise):
        """Simulate physical movement with motor noise."""
        std = np.sqrt(move_noise)
        actual_move = move_dist + np.random.normal(0, std)
        self.actual_x += actual_move
        return self.actual_x

    def get_ultrasonic_reading(self, sensor_noise):
        """Returns distance to the wall with sensor noise."""
        actual_dist = self.wall_pos - self.actual_x
        std = np.sqrt(sensor_noise)
        return actual_dist + np.random.normal(0, std)
