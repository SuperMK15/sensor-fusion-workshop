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
*Status: [COMING SOON]*
- Moving from discrete grids to Continuous Gaussians.
- **Concepts:** State Transition Matrix ($F$), Control Matrix ($B$), Kalman Gain ($K$). Read the theory in [2_kalman_filter/2_Kalman_Filter_Theory.pdf](./2_kalman_filter/2_Kalman_Filter_Theory.pdf)
- **Run:** `python 2_kalman_filter/main.py`
- **View:** Plots in `2_kalman_filter/plots`, each run gets its own timestamped plot.
- **Edit:** Constants in the `CONFIG` section of the [`2_kalman_filter/main.py`](./2_kalman_filter/main.py) code will change the behaviour of the filter.

### 3. Extended Kalman Filter (EKF)
*Status: [COMING SOON]*
- Handling non-linear motion and sensor models using Taylor Series expansion.
- **Concepts:** Jacobians, Linearization, Real-world noise models.
