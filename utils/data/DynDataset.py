import pickle
import torch
from torch.utils.data import Dataset

class DynDataset(Dataset):

    def __init__(self, file_path, device, num_traj=-1, data_type="point", split="train", keys=[]):

        self.device = device

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        self.num_traj = num_traj
        self.traj_len = len(data["t"][0])

        split_idx = {
            "train": (0, int(0.8 * self.num_traj)),
            "val": (int(0.8 * self.num_traj), int(0.9 * self.num_traj)),
            "test": (int(0.9 * self.num_traj), self.num_traj)
        }

        self.num_traj_split = split_idx[split][1] - split_idx[split][0]

        self.keys = keys
        self.data = { data_key: torch.tensor(data[data_key][split_idx[split][0]:split_idx[split][1]]).to(device).float() for data_key in keys }

        self.data_type = data_type
        self.n_data_point = self.traj_len * self.num_traj_split if data_type == "point" else self.num_traj_split
        self.manifold_dim = self.data["q"].shape[-1]

    def __len__(self):
        return self.n_data_point
    
    def __getitem__(self, idx):
        if self.data_type == "point":
            traj_idx = idx // self.traj_len
            point_idx = idx % self.traj_len
            return [self.data[key][traj_idx][point_idx] for key in self.keys]
        else:
            return [self.data[key][idx] for key in self.keys]