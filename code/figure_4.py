from model import ode_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl

# 10 different values of V4
V4s = np.linspace(0.1, 10., 10)
# one different color from a colormap for each V4
cmap = mpl.colormaps['plasma']
colors = cmap(np.linspace(0, 1, 10))

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

fig, axs = plt.subplots(2)

# solve ODEs for each V4 and plot P over time
for i, V4 in enumerate(V4s):
    param_dict['V4'] = V4
    sol  = solve_ivp(ode_model, [0, 140], [P_0, Q_0, R_0], method='RK23', args=(param_dict,),dense_output=True)
    # plot whole solution
    axs[0].plot(sol.t, sol.y[0], label = f'V4={np.round(V4,1)}', color=colors[i])
    # plot min and max of P
    axs[1].scatter([i+1,i+1], [min(sol.y[0]), max(sol.y[0])], color=colors[i])

axs[0].set(xlabel='Time', ylabel='P')
axs[1].set_xticks(range(1,11), [np.round(v,1) for v in V4s], rotation=45)
axs[1].set(xlabel='V4', ylabel='Min/Max P')
# big legend -> outside of plot
axs[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig('figure_4_different_V4.png')