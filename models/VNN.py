from torch import nn
from models.DifferentiableLayers import DifferentialLinear, DifferentialSoftplus

class VNN(nn.Module):
    def __init__(self, n_manifold):
        super().__init__()
        self.n_manifold = n_manifold
        self.MLP = nn.Sequential(
            DifferentialLinear(n_manifold, 128),
            DifferentialSoftplus(),
            DifferentialLinear(128, 128),
            DifferentialSoftplus(),
            DifferentialLinear(128, 1)
        )
    
    def forward(self, input):
        return self.MLP(input)
