from torch import nn
from models.MNN import MNN
from models.VNN import VNN

class HNN(nn.Module):

    def __init__(self, n_manifold):
        super().__init__()
        
        self.n_manifold = n_manifold
        
        #vectorized version of the mass-inertia matrix
        self.MNN = MNN(n_manifold)

        #potential energy
        self.VNN = VNN(n_manifold)

    def forward(self, q, dq, tau):
        