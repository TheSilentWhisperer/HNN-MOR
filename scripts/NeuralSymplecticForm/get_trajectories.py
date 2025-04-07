import numpy as np
import matplotlib.pyplot as plt
from utils.plot import PCA

def plot_trajectory(t, u, u_pred, config):
    save_path = f"runs/{config['RUN_ID']}/trajectory_plot.png"

    # PCA to reduce dimensions
    x = PCA(u)
    u_proj = u @ x
    u_pred_proj = u_pred @ x

    # plot with fading colors as time progresses
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))

    plt.scatter(u_proj[:, 0], u_proj[:, 1], label="True Trajectory", color="red", marker=".")
    plt.scatter(u_pred_proj[:, 0], u_pred_proj[:, 1], c=colors, label="Predicted Trajectory", marker=".")
    plt.colorbar(label="Time")
    plt.title("PCA projection of a trajectory on the manifold")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    import torch
    import json
    from utils.data.DynDataset import DynDataset
    from models.NeuralSymplecticForm import NeuralSymplecticForm
    from utils.integration import RK4
    import os

    # torch.manual_seed(0)

    config = json.load(open("scripts/NeuralSymplecticForm/config.json"))
    run_id = config["RUN_ID"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = DynDataset(config["DATA_PATH"], device, num_traj=config["NUM_TRAJ"], data_type="traj", split="test", keys=["t", "q", "dq"])

    model = NeuralSymplecticForm(dataset.manifold_dim).to(device)
    model.load_state_dict(torch.load(os.path.join("runs", run_id, "model.pt")))

    traj_idx = torch.randint(0, len(dataset), (1,)).item()
    t, q, dq = dataset[traj_idx]
    #take the first 100 points
    u = torch.cat((q, dq), dim=-1)

    # Get the predicted trajectory with RK4
    u_pred = RK4(model, u[0], t)
    plot_trajectory(t, u.cpu(), u_pred.cpu(), config)

