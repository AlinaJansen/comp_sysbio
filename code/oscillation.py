import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from model import ode_model


# Model parameters (plausible values chosen to produce oscillations)
param_dict = {
    'a': 1.0,
    'b': 0.8,
    'K': 0.1,
    'V1': 0.9,
    'V2': 1.2,
    'K1': 0.1,
    'K2': 0.1,
    'V3': 1.5,
    'V4': 0.6,
    'K3': 0.1,
    'K4': 0.1
}

# Initial conditions (in [0,1] range)
y0 = [0.2, 0.5, 0.1]

# Time span for integration
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 4000)

sol = solve_ivp(ode_model, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9, args=(param_dict,))

# Plot time series (single figure)
plt.figure(figsize=(10,5))
plt.plot(sol.t, sol.y[0], label='P (excess weight)')
plt.plot(sol.t, sol.y[1], label='Q (intake)')
plt.plot(sol.t, sol.y[2], label='R (restraint)')
plt.xlabel('Time')
plt.ylabel('State variables (dimensionless)')
plt.title('Model time series (P, Q, R) — example reproduction of Fig.2-like oscillations')
plt.legend()
plt.grid(True)
plt.savefig('time_series_P_Q_R.png')

# Plot phase-plane P vs R (separate figure)
plt.figure(figsize=(6,6))
plt.plot(sol.y[0], sol.y[2])
plt.xlabel('P (excess weight)')
plt.ylabel('R (restraint)')
plt.title('Phase plane: P vs R')
plt.grid(True)
plt.savefig('phase_plane_P_vs_R.png')
