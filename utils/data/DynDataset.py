import pickle
import torch
from torch.utils.data import Dataset

class DynDataset(Dataset):

    def __init__(self, file_path, device, data_type="point"):

        self.device = device

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        self.nb_traj = len(data["t"])
        self.traj_len = len(data["t"][0])

        self.q = torch.tensor(data["q"], dtype=torch.float32).to(device)
        self.dq = torch.tensor(data["dq"], dtype=torch.float32).to(device)
        self.ddq = torch.tensor(data["ddq"], dtype=torch.float32).to(device)

        self.data_type = data_type
        self.n_data_point = self.traj_len * self.nb_traj if data_type == "point" else self.nb_traj
        self.manifold_dim = self.q.shape[-1] * 2

    def __len__(self):
        return self.n_data_point
    
    def __getitem__(self, idx):
        if self.data_type == "point":
            traj_idx = idx // self.traj_len
            point_idx = idx % self.traj_len
            return (
                self.q[traj_idx, point_idx],
                self.dq[traj_idx, point_idx],
                self.ddq[traj_idx, point_idx]
            )
        else:
            return (
                self.q[idx],
                self.dq[idx],
                self.ddq[idx]
            )