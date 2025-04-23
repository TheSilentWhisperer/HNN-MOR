import torch
from torch import nn
import torch.nn.functional as F

class DifferentialLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        #pytorch default initialization
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # xavier_normal_ initialization
        # nn.init.xavier_normal_(self.weight, gain = np.sqrt(2. / (1 + np.pi ** 2 / 6)))
        
    def forward(self, input):
        x, dxdq = input["x"], input["dxdq"]
        x = F.linear(x, self.weight, self.bias)
        dxdq = self.weight @ dxdq
        return {"x": x, "dxdq": dxdq}

class DifferentialSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x, dxdq = input["x"], input["dxdq"]
        jac = torch.diag_embed(torch.sigmoid(x))
        x = F.softplus(x)
        dxdq = jac @ dxdq
        return {"x": x, "dxdq": dxdq}