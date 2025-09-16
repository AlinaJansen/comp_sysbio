from model import ode_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl

# 10 different values for all K_i
Ks = np.linspace(0.01, 0.1, 10)
# one different color from a colormap for each posssible K_i
cmap = mpl.colormaps['plasma']
colors = cmap(np.linspace(0, 1, 10))

# Initial conditions
P_0 = 0.5
Q_0 = 0.8
R_0 = 0.8

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

plt.figure()

# solve ODEs for K1 = K2 = K3 = K4 for different values and plot P over time
for i, K in enumerate(Ks):
    param_dict['K1'] = K
    param_dict['K2'] = K
    param_dict['K3'] = K
    param_dict['K4'] = K
    sol  = solve_ivp(ode_model, [0, 50], [P_0, Q_0, R_0], method='RK23', args=(param_dict,),dense_output=True)
    plt.plot(sol.t, sol.y[0], label = f'K_i={np.round(K,2)}', color=colors[i])

# big legend -> outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time')
plt.ylabel('P', rotation=0)
plt.tight_layout()
plt.savefig('figure_6_different_K_i.png')