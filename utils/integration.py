import torch
from tqdm import tqdm

def RK4(model, u0, tau, t):
    
    u = [u0]

    for i in tqdm(range(1, len(t))):
        h = t[i] - t[i-1]
        k1 = model.vect_field(u[i-1], tau[i-1])
        k2 = model.vect_field(u[i-1] + h / 2 * k1, tau[i-1])
        k3 = model.vect_field(u[i-1] + h / 2 * k2, tau[i-1])
        k4 = model.vect_field(u[i-1] + h * k3, tau[i-1])
        u.append(u[i-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    return torch.stack(u)