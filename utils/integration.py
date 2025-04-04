import torch

def RK4(model, y0, t):
    y = [y0]
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        k1 = model.predict(y[i])
        k2 = model.predict(y[i] + h / 2 * k1)
        k3 = model.predict(y[i] + h / 2 * k2)
        k4 = model.predict(y[i] + h * k3)
        y_next = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y.append(y_next)

    return torch.stack(y)