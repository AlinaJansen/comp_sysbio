from model import ode_model, jacobian_at

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Find steady state ----
def find_sstate(initial_guess, params):
    func = lambda y: ode_model(0, y, params)
    ystar, _, ier, mesg = fsolve(func, initial_guess, full_output=True)
    if ier != 1:
        raise RuntimeError("fsolve did not converge: " + mesg)
    return ystar

# ---- Scan one parameter for stability and steady states ----
def scan_param(param_name, param_values, initial_guess, params_base):
    assert param_name in params_base, "Given parameter name does not exist in given parameter dictionary."

    max_real = []
    sstates = []
    y_guess = initial_guess.copy()
    params = params_base.copy()
    for val in param_values:
        params[param_name] = val
        # Find steady state
        try:
            ystar = find_sstate(y_guess, params)
        except RuntimeError:
            # If fsolve did not converge, use initial guess again
            ystar = find_sstate(initial_guess, params)
        sstates.append(ystar)

        # Compute eigenvalues of Jacobian at steady state
        J = jacobian_at(ystar, params)
        eigvals = np.linalg.eigvals(J)

        # Save max real part of eigenvalues
        # > 0 -> unstable, < 0 -> stable, = 0 -> bifurcation
        max_real.append(np.max(np.real(eigvals)))

        # use previous steady state as next initial guess
        y_guess = ystar

    return np.array(sstates), np.array(max_real)

# Initial conditions
P_0 = 0.25
Q_0 = 0.9
R_0 = 0.02

# Parameters
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

# 10 different values of V4
V4s = np.linspace(0.1, 10., 10)

sstates, max_reals = scan_param('V4', V4s, [P_0, Q_0, R_0], param_dict)

fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0.1)
axs = gs.subplots(sharex=True)

for i, V4 in enumerate(V4s):
    param_dict['V4'] = V4
    sol  = solve_ivp(ode_model, [0, 140], [P_0, Q_0, R_0], method='RK23', args=(param_dict,),dense_output=True)
    # plot min and max of P
    axs[0].scatter(i+1, min(sol.y[0]), color='black', marker='^', s=8)
    axs[0].scatter(i+1, max(sol.y[0]), color='black', marker='v', s=8)
    axs[0].scatter(i+1, sstates[i,0], color='black', marker='*', s=10) # steady state
    axs[0].set(xlabel='V4', ylabel='P')
    # plot min and max of Q
    axs[1].scatter(i+1, min(sol.y[1]), color='black', marker='^', s=8)
    axs[1].scatter(i+1, max(sol.y[1]), color='black', marker='v', s=8)
    axs[1].scatter(i+1, sstates[i,1], color='black', marker='*', s=10) # steady state
    axs[1].set(xlabel='V4', ylabel='Q')
    # plot min and max of R
    axs[2].scatter(i+1, min(sol.y[2]), color='black', marker='^', s=8)
    axs[2].scatter(i+1, max(sol.y[2]), color='black', marker='v', s=8)
    axs[2].scatter(i+1, sstates[i,2], color='black', marker='*', s=10) # steady state
    axs[2].set(xlabel='V4', ylabel='R')
for i in range(3):
    axs[i].set_xticks(range(1,11), [np.round(v,1) for v in V4s])

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

# build custom legend
legend_elems = [Line2D([0], [0], marker='^', label='min', markersize=8, color='w', markerfacecolor='black'),
                Line2D([0], [0], marker='*', label='sstate', markersize=10, color='w', markerfacecolor='black'),
                Line2D([0], [0], marker='v', label='max', markersize=8, color='w', markerfacecolor='black')]
axs[0].legend(handles=legend_elems, loc='upper left')

fig.savefig('sstate_analysis_V4_figure.png')