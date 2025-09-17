import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from model import ode_model


# Model parameters
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

# initial conditions estimated from fig 3A in paper
y0 = [0.43, 0.725, 0.55]

# Time span for integration
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 4000)

def model_task_3(integration_method, dot = None):
    sol = solve_ivp(ode_model, t_span, y0, method=integration_method, t_eval=t_eval, rtol=1e-6, atol=1e-9, args=(param_dict,))

    plt.figure(figsize=(10,5))
    plt.plot(sol.t, sol.y[0], label='P (excess weight)')
    plt.plot(sol.t, sol.y[1], label='Q (intake)')
    plt.plot(sol.t, sol.y[2], label='R (restraint)')
    plt.xlabel('Time')
    plt.ylabel('State variables (dimensionless)')
    plt.title('Model time series (P, Q, R) — example reproduction of Fig.2-like oscillations')
    plt.legend()
    plt.title(f'{integration_method}')
    plt.grid(True)
    plt.savefig(f'figures/fig_3_a_timecourse_{integration_method}.png', dpi=300)

    '''
    Creating figure 3B
    '''

    # initial values
    y0_3B = [[0.43, 0.8, 0.05], [0.43, 0.9, 0.6]]

    solutions = []

    for initial_y in y0_3B:
        sol = solve_ivp(ode_model, t_span, initial_y, method=integration_method, t_eval=t_eval, rtol=1e-6, atol=1e-9, args=(param_dict,))
        solutions.append(sol)

    plt.figure(figsize=(6,6))
    for num, sol in enumerate(solutions):
        plt.plot(sol.y[2], sol.y[0], label=f'IC={y0_3B[num][0]:.2f}, {y0_3B[num][1]:.2f}, {y0_3B[num][2]:.2f}')
        if dot:
           plt.plot(dot[2], dot[0], 'ro')

    plt.xlabel('Cognitive restraint, R')
    plt.ylabel('Weight, P')
    plt.grid(True)
    plt.legend()
    if dot:
        plt.savefig(f'figures/fig_3_b_phaseplane_{integration_method}_dot.png')
    else:
        plt.savefig(f'figures/fig_3_b_phaseplane_{integration_method}.png')

available_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

for method in available_methods:
    model_task_3(method)