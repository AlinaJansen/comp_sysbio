import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from model import ode_model

colors = ['red', 'blue']

param_dict = {
    'a': 0.1,
    'b': 0.1,
    'K': 0.2,
    'V1': 1.0,
    'V2': 1.5,
    'K1': 0.01,
    'K2': 0.01,
    'V3': 6.0,
    'V4': 2.5,
    'K3': 0.01,
    'K4': 0.01
}

# Definitions of nullclines / steady-state functions

def R_ss(P):
    """
    Solve steady state of dR/dt = 0 for R given P:
      V3 * P * (1 - R)/(K3 + (1 - R)) - V4 * R/(K4 + R) = 0
    Returns R in [0,1]
    """
    # Define function whose root is steady-state R
    def f(R):
        return V3 * P * (1 - R)/(K3 + (1 - R)) - V4 * R/(K4 + R)
    # Provide an initial guess for R
    # We can try two guesses and pick the one in [0,1]
    R_guess = 1.0
    R_solution, info, ier, mesg = fsolve(f, R_guess, full_output=True)
    # # Clamping between 0 and 1
    # R_solution = np.clip(R_solution, 0.0, 1.0)
    return R_solution, ier

def Q_ss(R):
    """
    Solve steady state of dQ/dt = 0 for Q given R:
      V1 * (1 - Q)/(K1 + (1 - Q)) - V2 * R * Q/(K2 + Q) = 0
    Returns Q in [0,1]
    """
    def f(Q):
        return V1 * (1 - Q)/(K1 + (1 - Q)) - V2 * R * Q/(K2 + Q)
    Q_guess = 1
    Q_solution, info, ier, mesg = fsolve(f, Q_guess, full_output=True)
    Q_solution = np.clip(Q_solution, 0.0, 1.0)
    return Q_solution, ier

# Prepare P grid, compute R_ss for each
P_vals = np.linspace(0, 1, 500)
R_vals = []
solved_R = []
for p in P_vals:
    r, ier = R_ss(p)
    R_vals.append(r)
    solved_R.append(ier)
solved_R = [1 if i == 1 else 0 for i in solved_R]

# Prepare R grid, compute Q_ss for each
R_grid = np.linspace(0, 1, 500)
Q_vals = []
solved_Q = []
for p in P_vals:
    q, ier = Q_ss(p)
    Q_vals.append(q)
    solved_Q.append(ier)
solved_Q = [1 if i == 1 else 0 for i in solved_Q]

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].scatter(P_vals, R_vals, color=[colors[i] for i in solved_R], label='Steady-state R vs P')
axes[0].set_xlabel('P (excess weight, normalized)')
axes[0].set_ylabel('R (cognitive restraint, steady-state)')
axes[0].set_title('Threshold: R vs P')
axes[0].grid(True)

qcolor = [colors[i] for i in solved_Q]
axes[1].scatter(R_grid, Q_vals, color=qcolor, label='Steady-state Q vs R')
axes[1].set_xlabel('R (cognitive restraint, normalized)')
axes[1].set_ylabel('Q (intake, steady-state)')
axes[1].set_title('Threshold: Q vs R')
axes[1].grid(True)

plt.show('figure_2_thresholds.png')