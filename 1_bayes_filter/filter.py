import numpy as np
from scipy.stats import norm

class DiscreteBayesFilter:
    def __init__(self, grid_size, map_range):
        """
        Initializes the belief PDF as a discrete grid.
        Matches Section 2 of the notes (The Variables).
        """
        self.nodes = np.linspace(0, map_range, grid_size)
        # Initial belief: Uniform distribution (complete ignorance)
        self.bel = np.ones(grid_size) / grid_size

    def predict(self, u, move_noise):
        """
        Step A: The Prediction (Motion Update) - See Section 3.A
        Calculates the 'Prior Belief' (bar_bel) by shifting/blurring the cloud.
        """
        prior_bel = np.zeros_like(self.bel)
        std = np.sqrt(move_noise)
        
        # Implementation of the Integral from the notes:
        # Summing on all points in the cloud x_{t-1} as launchpads
        for i in range(len(self.bel)):
            if self.bel[i] > 1e-4:
                # P(x_t | u_t, x_{t-1}): The Motion Model
                transition_pdf = norm.pdf(self.nodes, loc=self.nodes[i] + u, scale=std)
                # Weighting the transition by our current belief at that launchpad
                prior_bel += transition_pdf * self.bel[i]
        
        # Ensure the prior is a valid PDF (sum to 1)
        self.bel = prior_bel / np.sum(prior_bel)

    def update(self, z_dist, sensor_noise, wall_pos):
        """
        Step B: The Correction (Measurement Update) - See Section 3.B
        Calculates the 'Posterior Belief' by sharpening the blurry guess.
        """
        # P(z_t | x_t): The Sensor Model / Likelihood
        # Calculated by comparing sensed distance to expected distance (wall - nodes)
        expected_z = wall_pos - self.nodes
        likelihood = norm.pdf(z_dist, loc=expected_z, scale=np.sqrt(sensor_noise))
        
        # Bel(x_t) = n * P(z_t | x_t) * bar_Bel(x_t)
        # Point-by-point multiplication (Information Fusion - Section 4)
        self.bel = likelihood * self.bel
        
        # Step C: The n (Eta) Scaling Factor
        # Normalizes the "shrunken" PDF back to an area of 1.0
        eta = 1.0 / np.sum(self.bel)
        self.bel *= eta

    def get_stats(self):
        """
        Extracts point estimates from the PDF cloud.
        Useful for comparing the "Mean" to the "True Pos" in Section 5.
        """
        mean = np.sum(self.nodes * self.bel)
        variance = np.sum(self.bel * (self.nodes - mean)**2)
        return mean, variance
