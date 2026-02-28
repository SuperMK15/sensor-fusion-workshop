# Sensor Fusion Workshop

This repository provides a step-by-step implementation of sensor fusion algorithms, moving from theoretical foundations to real-world robotics applications.

## 🚀 Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/SuperMK15/sensor_fusion_workshop.git
   cd sensor_fusion_workshop
   ```

2. **Create and activate a virtual envrionment:**
   ```bash
   # Create the venv
   python -m venv venv

   # Activate it (Windows)
   .\venv\Scripts\activate

   # Activate it (Mac/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install .
   ```

## 📂 Modules

### 1. Bayes Filter (Discrete)
The foundation of all estimation. Uses a discrete grid (histogram) to represent the probability distribution. 
- **Concepts:** Prediction (Motion Update), Correction (Measurement Update), Normalization. Read the theory in [1_bayes_filter/1_Bayes_Filter_Theory.pdf](./1_bayes_filter/1_Bayes_Filter_Theory.pdf).
- **Run:** `python 1_bayes_filter/main.py`
- **View:** Plots in `1_bayes_filter/plots`, each run gets its own timestamped plot.
- **Edit:** Constants in the `CONFIG` section of the [`1_bayes_filter/main.py`](./1_bayes_filter/main.py) code will change the behaviour of the filter.

### 2. Kalman Filter (Linear)
Optimal state estimation for linear systems with Gaussian noise, using recursive prediction and correction.
- **Concepts:** Linear system modeling, State Transition Matrix ($F$), Control Matrix ($B$), Process Noise ($Q$), Measurement Matrix ($H$), Measurement Noise ($R$), Covariance Propagation, Kalman Gain ($K$), Innovation (Residual). Read the theory in [2_kalman_filter/2_Kalman_Filter_Theory.pdf](./2_kalman_filter/2_Kalman_Filter_Theory.pdf)
- **Run:** `python 2_kalman_filter/main.py`
- **View:** Plots in `2_kalman_filter/plots`, each run gets its own timestamped plot.
- **Edit:** Constants in the `CONFIG` section of the [`2_kalman_filter/main.py`](./2_kalman_filter/main.py) code will change the behaviour of the filter.

### 3. Extended Kalman Filter (EKF)
Nonlinear state estimation by locally linearizing system dynamics and measurements with Taylor Series and Jacobians.
- **Concepts:** Nonlinear state transition $f(x,u)$, Nonlinear measurement model $h(x)$, Jacobians ($F_t = \frac{\partial f}{\partial x}$, $H_t = \frac{\partial h}{\partial x}$), First-order Taylor linearization, Covariance propagation, Innovation, Additive vs Multiplicative error states, Realistic sensor noise modeling. Read the theory in [3_ekf/3-1_EKF_Theory.pdf](./3_ekf/3-1_EKF_Theory.pdf) and [3_ekf/3-2_MEKF_Theory.pdf](./3_ekf/3-2_MEKF_Theory.pdf)
- **Run:** `python 3_ekf/main.py` (can pass `--filter aekf` or `--filter mekf` to switch between Additive and Multiplicative EKFs, default is AEKF and also `--sim-mode oscillate` or `--sim-mode const_rot`, default is oscillate)
- **View:** Plots in `3_ekf/plots`, each run get its own timestamped plot.
- **Edit:** Constants in the `CONFIG` section of the [`3_ekf/main.py`](./3_ekf/main.py) code will change the behaviour of the filter.
