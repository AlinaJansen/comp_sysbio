def ode_model(t, y, param_dict):
    P, Q, R = y
    dP = param_dict['a']*Q - param_dict['b'] * P / (param_dict['K'] + P)
    dQ = param_dict['V1'] * (1 - Q) / (param_dict['K1'] + (1 - Q)) - param_dict['V2'] * R * Q / (param_dict['K2'] + Q)
    dR = param_dict['V3'] * P * (1 - R) / (param_dict['K3'] + (1 - R)) - param_dict['V4'] * R / (param_dict['K4'] + R)
    return [dP, dQ, dR]