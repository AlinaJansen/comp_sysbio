import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from model import ode_model_wo_time
from scipy.optimize import fsolve
from model import ode_model

def model_task_3(integration_method, dot = None, eigenvectors = None, eigenvalues = None):
    '''
    Creating figure 3B
    '''
    t_span = (0, 40)
    t_eval = np.linspace(t_span[0], t_span[1], 4000)
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
           plt.plot(dot[2], dot[0], 'ro', zorder=0)

    for num, ev in enumerate(eigenvectors):
        arrow_coors = [(dot[2], dot[0])]
        arrow_coors.append(ev*.02)
        if eigenvalues[num] > 0:
            pass
        else:
            arrow_coors[0] = arrow_coors[0]+ arrow_coors[1]
            arrow_coors[1] = -arrow_coors[1]
        plt.arrow(arrow_coors[0][0], arrow_coors[0][1], arrow_coors[1][0], arrow_coors[1][1], head_width = 0.008, length_includes_head=True, head_length= 0.001)
    plt.xlabel('Cognitive restraint, R')
    plt.ylabel('Weight, P')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'figures/fig_3_b_phaseplane_{integration_method}_dot_vec_mini.png', dpi=300)

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

# initial guess
y0 = [0.43, 0.725, 0.55]

root, info, ier, mesg = fsolve(ode_model_wo_time, y0, args=(param_dict), full_output=True)

# root:
# array([0.42234428, 0.67863447, 0.65607509])

R_mat = np.array([[info['r'][0], info['r'][1], info['r'][2]],[0, info['r'][3], info['r'][4]],[0, 0, info['r'][5]]])
eigenvalues, eigenvectors = np.linalg.eig(info['fjac']*R_mat)
eigenvectors_split = [eigenvectors[[2,0],i] for i in range(eigenvectors.shape[1])]

model_task_3('RK45', dot = list(root), eigenvalues=eigenvalues, eigenvectors=eigenvectors_split)
pass