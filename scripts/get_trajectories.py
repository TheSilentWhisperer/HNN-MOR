import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch

def plot_trajectories(trajectories, t):
    colors = {
        "Ground Truth": "black",
        "LNN": "blue",
        "BB": "orange"
    }

    t = t.detach().cpu().numpy()
    n_manifold = trajectories["Ground Truth"].shape[-1] // 2

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    for key, u in trajectories.items():
        q, dq = torch.split(u, n_manifold, dim=-1)
        q1 = q[:, 0].detach().cpu().numpy()
        q2 = q[:, 1].detach().cpu().numpy()
        dq1 = dq[:, 0].detach().cpu().numpy()
        dq2 = dq[:, 1].detach().cpu().numpy()

        ax[0, 0].plot(t, q1, label=key, color=colors[key])
        ax[0, 1].plot(t, q2, label=key, color=colors[key])
        ax[1, 0].plot(t, dq1, label=key, color=colors[key])
        ax[1, 1].plot(t, dq2, label=key, color=colors[key])

    ax[0, 0].set_title("q1")
    ax[0, 1].set_title("q2")
    ax[1, 0].set_title("dq1")
    ax[1, 1].set_title("dq2")

    # Create a common legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(colors))

    plt.tight_layout()
    plt.savefig("results/trajectories.png")
    plt.show()

    plt.close()


def plot_energies(energies, t):
    
    colors = {
        "Ground Truth": "black",
        "LNN": "blue"
    }

    t = t.detach().cpu().numpy()

    fig, ax = plt.subplots(3, 1)

    for key, energy in energies.items():
        ax[0].plot(t, energy[:, 0].detach().cpu().numpy(), label=key, color=colors[key])
        ax[1].plot(t, energy[:, 1].detach().cpu().numpy(), label=key, color=colors[key])
        ax[2].plot(t, energy[:, 2].detach().cpu().numpy(), label=key, color=colors[key])

    ax[0].set_title("Kinetic Energy")
    ax[1].set_title("Potential Energy")
    ax[2].set_title("Total Energy")
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(colors))

    plt.tight_layout()
    plt.savefig("results/energies.png")
    plt.show()

    
if __name__ == "__main__":
    import torch
    import json
    from utils.data.DynDataset import DynDataset
    from models.LNN import LNN
    from models.BB import BB
    from utils.integration import RK4
    import os

    # torch.manual_seed(0)

    data_config = json.load(open("scripts/data_config.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = DynDataset(data_config["DATA_PATH"], device, num_traj=data_config["NUM_TRAJ"], data_type="traj", split="test", keys=["t", "q", "dq", "tau", "e_pot", "e_kin", "m"])

    traj_idx = torch.randint(0, len(dataset), (1,)).item()
    t, q, dq, tau, e_pot, e_kin, m = dataset[traj_idx]

    t = t[:3000]
    q = q[:3000]
    dq = dq[:3000]
    tau = tau[:3000]
    e_pot = e_pot[:3000]
    e_kin = e_kin[:3000]

    e_tot = e_pot + e_kin
    e = torch.stack((e_kin, e_pot, e_tot), dim=-1)
    u = torch.cat((q, dq), dim=-1)

    n_manifold = u.shape[-1] // 2

    models = {
        "LNN": LNN(n_manifold).to(device),
        "BB": BB(n_manifold).to(device)
    }

    trajectories = {
        "Ground Truth": u
    }

    for key in models.keys():
        config = json.load(open(f"scripts/{key}/config.json"))
        run_id = config["RUN_ID"]
        models[key].load_state_dict(torch.load(os.path.join("runs", run_id, "model.pt"))) 

        # Get the predicted trajectory with RK4
        u_pred = RK4(models[key], u[0], tau, t)
        trajectories[key] = u_pred


plot_trajectories(trajectories, t)

energies = {
    "Ground Truth": e,
    "LNN": models["LNN"].get_energies(u_pred)
}
plot_energies(energies, t)