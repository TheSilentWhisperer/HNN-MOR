import torch
from torch import nn

class BB(nn.Module):

    def __init__(self, n_manifold):
        super().__init__()
        self.n_manifold = n_manifold
        self.MLP = nn.Sequential(
            nn.Linear(3*n_manifold, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_manifold)
        )

    def forward(self, q, dq, tau):
        u = torch.cat((q, dq, tau), dim=-1)
        ddq = self.MLP(u)
        return ddq
    
    def vect_field(self, u, tau):
        q, dq = torch.split(u.view(1, -1), self.n_manifold, dim=-1)
        tau = tau.view(1, -1)
        ddq = self.forward(q, dq, tau)
        return torch.cat((dq, ddq), dim=-1).view(-1)
