import os
import matplotlib.pyplot as plt
from datetime import datetime
from simulation import RobotSimulation
from filter import DiscreteBayesFilter

# --- CONFIG ---
NUM_STEPS = 5 # Number of time steps to simulate
MOVE_DIST = 1.0 # Nominal movement distance per step (meters)
MOVE_NOISE = 0.2 # Standard deviation of movement noise (meters)
SENSOR_NOISE = 0.1 # Standard deviation of sensor noise (meters)
START_POS = 2.0 # Starting position of the robot (meters)
WALL_POS = 12.0 # Position of the wall measured by the ultrasonic sensor (meters)
GRID_RES = 500 # Number of discrete points in the grid for the Bayes filter
# --- CONFIG ---

def run_demo():
    # Setup directories
    plot_dir = os.path.join("1_bayes_filter", "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    sim = RobotSimulation(START_POS, WALL_POS)
    bf = DiscreteBayesFilter(GRID_RES, WALL_POS)
    
    # --- PREPARE PLOT ---
    # Create a vertical stack of subplots (one for each step)
    fig, axes = plt.subplots(NUM_STEPS, 1, figsize=(10, 3 * NUM_STEPS), sharex=True)
    if NUM_STEPS == 1: axes = [axes] # Handle single step case
    
    print(f"{'Step':<4} | {'True X':<8} | {'Prior (μ, σ²)':<18} | {'Post (μ, σ²)':<18}")
    print("-" * 65)

    for i in range(1, NUM_STEPS + 1):
        ax = axes[i-1]
        
        # 1. PREDICT
        true_x = sim.move(MOVE_DIST, MOVE_NOISE)
        bf.predict(MOVE_DIST, MOVE_NOISE)
        prior_mu, prior_var = bf.get_stats()
        
        # Plot Prior (The guess)
        ax.fill_between(bf.nodes, bf.bel, color='red', alpha=0.2, label='Prior (Guess)')

        # 2. UPDATE
        z = sim.get_ultrasonic_reading(SENSOR_NOISE)
        bf.update(z, SENSOR_NOISE, WALL_POS)
        post_mu, post_var = bf.get_stats()

        # Plot Posterior (The result)
        ax.plot(bf.nodes, bf.bel, color='green', linewidth=2, label=f'Step {i} Posterior')
        ax.fill_between(bf.nodes, bf.bel, color='green', alpha=0.3)

        # Plot Reference Lines
        ax.axvline(true_x, color='black', linestyle='--', label='True Position')
        ax.axvline(WALL_POS, color='blue', linewidth=2, label='Wall')
        
        # Formatting for this specific step
        ax.set_title(f"Step {i}: True Pos = {true_x:.2f} m | Prior μ={prior_mu:.2f}, σ²={prior_var:.2f} | Post μ={post_mu:.2f}, σ²={post_var:.2f}")
        ax.set_ylabel("Prob Density")
        if i == 1: ax.legend(loc='upper left')

        print(f"{i:<4} | {true_x:<8.2f} | {prior_mu:>5.2f}, {prior_var:>5.2f}    | {post_mu:>5.2f}, {post_var:>5.2f}")

    plt.xlabel("Position in Hallway (m)")
    plt.tight_layout()

    # Save the graph with a timestamp to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plot_dir, f"bayes_filter_{timestamp}.png")
    plt.savefig(save_path)
    print(f"\nGraph saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_demo()
