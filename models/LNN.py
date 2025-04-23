import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from models.DifferentiableLayers import DifferentialLinear, DifferentialSoftplus
from models.MNN import MNN
from models.VNN import VNN

class LNN(nn.Module):
    """
    Takes a point (q, dq) in the manifold and tau(t) and returns the acceleration ddq according to the Lagrangian dynamics.
    """
    def __init__(self, n_manifold):
        super().__init__()
        
        self.n_manifold = n_manifold
        
        #vectorized version of the mass-inertia matrix
        self.MNN = MNN(n_manifold)

        #potential energy
        self.VNN = VNN(n_manifold)

    def forward(self, q, dq, tau):
        """
        q shape: (batch_size, n_manifold)
        dq shape: (batch_size, n_manifold)
        tau shape: (batch_size, n_manifold)
        """

        batch_size, n = q.shape

        batch_eye = torch.eye(n).repeat(batch_size, 1, 1).to(q.device)
        Moutput = self.MNN({"x": q, "dxdq": batch_eye})
        M, dMdq = Moutput["x"], Moutput["dxdq"]

        dVdq = self.VNN({"x": q, "dxdq": batch_eye})["dxdq"]
        g = dVdq.squeeze(1)
    
        C = 0.5 * torch.einsum('bijk,bk->bij', (dMdq + dMdq.transpose(2, 3) - dMdq.transpose(1, 3)), dq)
        c = torch.einsum('bij,bj->bi', C, dq)

        #compute the acceleration ddq
        ddq = torch.linalg.solve(M, tau - c - g)
        return ddq
    
    def vect_field(self, u, tau):
        q, dq = torch.split(u.view(1, -1), self.n_manifold, dim=-1)
        ddq = self.forward(q, dq, tau)
        return torch.cat((dq, ddq), dim=-1).view(-1)
    
    def get_kinetic_energy(self, q, dq):
        batch_eye = torch.eye(self.n_manifold).unsqueeze(0).repeat(q.shape[0], 1, 1).to(q.device)
        M = self.MNN({"x": q, "dxdq": batch_eye})["x"]
        T = 0.5 * torch.einsum('bi, bij, bj -> b', dq, M, dq)
        return T
    
    def get_potential_energy(self, q):
        batch_eye = torch.eye(self.n_manifold).unsqueeze(0).repeat(q.shape[0], 1, 1).to(q.device)
        V = self.VNN({"x": q, "dxdq": batch_eye})["x"]
        return V.squeeze()
    
    def get_total_energy(self, T, V):
        return T + V
    
    def get_energies(self, u):
        q, dq = torch.split(u, self.n_manifold, dim=-1)
        T = self.get_kinetic_energy(q, dq)
        V = self.get_potential_energy(q)
        E = self.get_total_energy(T, V)
        return torch.stack((T, V, E), dim=-1)
    
if __name__ == "__main__":
    #test the MNN class
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_manifold = 2
    
    q = torch.randn(4, n_manifold).to(device)
    model = nn.Sequential(
        DifferentialLinear(n_manifold, 128),
        DifferentialSoftplus(),
        DifferentialLinear(128, 128),
        DifferentialSoftplus(),
        DifferentialLinear(128, n_manifold*(n_manifold+1)//2)    
    ).to(device)
    dqdq = torch.eye(n_manifold).unsqueeze(0).repeat(q.shape[0], 1, 1).to(device)
    output = model({"x": q, "dxdq": dqdq})
    def f(q):
        output = model({"x": q, "dxdq": dqdq})
        Uvect = output["x"]
        return Uvect
    jac = vmap(jacrev(f, 0))(q)
    handjac = output["dxdq"]
    print(jac-handjac)
