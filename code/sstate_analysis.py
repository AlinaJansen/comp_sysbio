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

# disables plotting, recommended for large scans
# stable steady states are always printed to console
disable_plotting = False

# parameter to scan and its values
param_name = 'V1'
param_values = np.linspace(0.01, .5, 25)
decimal_places = 3 # for printing parameter values and steady states

sstates, max_reals = scan_param(param_name, param_values, [P_0, Q_0, R_0], param_dict)
for state, p, i in zip(max_reals, param_values, range(len(param_values))):
    if state < 0:
        print(f'Stable steady state at {param_name}={np.round(p, decimal_places)}: P={np.round(sstates[i,0], decimal_places)}, Q={np.round(sstates[i,1], decimal_places)}, R={np.round(sstates[i,2], decimal_places)}')

if disable_plotting:
    exit(0)

# ---- Plot steady states and min/max values of variables during time course ----  
fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0.1)
axs = gs.subplots(sharex=True)

for i, p in enumerate(param_values):
    param_dict[param_name] = p
    sol  = solve_ivp(ode_model, [0, 800], [P_0, Q_0, R_0], method='RK23', args=(param_dict,),dense_output=True)
    for j in range(3):
        axs[j].scatter(i+1, min(sol.y[j]), color='black', marker='^', s=8)
        axs[j].scatter(i+1, max(sol.y[j]), color='black', marker='v', s=8)
        c = 'black' # stable
        if max_reals[i] > 0:
            c = 'red' # unstable
        axs[j].scatter(i+1, sstates[i,j], color=c, marker='*', s=10) # steady state

for i in range(3):
    axs[i].set_xticks(range(1,len(param_values)+1), [np.round(v,2) for v in param_values], rotation=45)

axs[0].set(xlabel=f'{param_name}', ylabel='P')
axs[1].set(xlabel=f'{param_name}', ylabel='Q')
axs[2].set(xlabel=f'{param_name}', ylabel='R')

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

# build custom legend
legend_elems = [Line2D([0], [0], marker='v', label='max', markersize=6, color='w', markerfacecolor='black'),
                Line2D([0], [0], marker='*', label='stable sstate', markersize=8, color='w', markerfacecolor='black'),
                Line2D([0], [0], marker='*', label='unstable sstate', markersize=8, color='w', markerfacecolor='red'),
                Line2D([0], [0], marker='^', label='min', markersize=6, color='w', markerfacecolor='black')]
axs[0].legend(handles=legend_elems, fontsize='x-small', loc='upper left')

fig.savefig(f'sstate_analysis_{param_name}_figure.png')