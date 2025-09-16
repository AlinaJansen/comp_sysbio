import numpy as np

def ode_model(t, y, param_dict):
    P, Q, R = y
    dP = param_dict['a']*Q - param_dict['b'] * P / (param_dict['K'] + P)
    dQ = param_dict['V1'] * (1 - Q) / (param_dict['K1'] + (1 - Q)) - param_dict['V2'] * R * Q / (param_dict['K2'] + Q)
    dR = param_dict['V3'] * P * (1 - R) / (param_dict['K3'] + (1 - R)) - param_dict['V4'] * R / (param_dict['K4'] + R)
    return [dP, dQ, dR]

def jacobian_at(y, param_dict):
    P, Q, R = y

    # unpack parameters for readability (python compiler should optimize this away)
    a = param_dict['a']; b = param_dict['b']; K = param_dict['K']
    V1 = param_dict['V1']; V2 = param_dict['V2']; V3 = param_dict['V3']; V4 = param_dict['V4']
    K1 = param_dict['K1']; K2 = param_dict['K2']; K3 = param_dict['K3']; K4 = param_dict['K4']
    
    # Partial derivatives
    f1_P = - b * K / (K + P)**2
    f1_Q = a
    f1_R = 0.0
    
    f2_Q = - V1 * K1 / (K1 + 1.0 - Q)**2 - V2 * R * K2 / (K2 + Q)**2
    f2_R = - V2 * Q / (K2 + Q)
    f2_P = 0.0
    
    f3_P = V3 * (1.0 - R) / (K3 + 1.0 - R)
    f3_R = - V3 * P * K3 / (K3 + 1.0 - R)**2 - V4 * K4 / (K4 + R)**2
    f3_Q = 0.0
    
    J = np.array([
        [f1_P, f1_Q, f1_R],
        [f2_P, f2_Q, f2_R],
        [f3_P, f3_Q, f3_R]
    ])
    return J