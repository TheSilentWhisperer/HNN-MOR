if __name__ == "__main__":
    import torch
    import json
    from utils.data.DynDataset import DynDataset
    from models.NeuralSymplecticForm import NeuralSymplecticForm
    from utils.integration import RK4
    from utils.plot import plot_trajectory
    import os

    # torch.manual_seed(0)

    config = json.load(open("scripts/config.json"))
    run_id = config["RUN_ID"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = DynDataset(config["DATA_PATH"], device, data_type="traj", split="test", keys=["t", "q", "dq"])

    model = NeuralSymplecticForm(dataset.manifold_dim).to(device)
    model.load_state_dict(torch.load(os.path.join("runs", run_id, "model.pt")))

    traj_idx = torch.randint(0, len(dataset), (1,)).item()
    t, q, dq = dataset[traj_idx]
    u = torch.cat((q, dq), dim=-1)

    # Get the predicted trajectory with RK4
    u_pred = RK4(model, u[0], t)
    q_pred = u_pred[:, :dataset.manifold_dim]
    plot_trajectory(t, q.cpu(), q_pred.cpu(), config)

