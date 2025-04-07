import torch
import torch.nn as nn
from torch.func import jacrev, vmap

class ClosenessEnforcer(nn.Module):
    def __init__(self, manifold_dim):
        super().__init__()
        self.manifold_dim = manifold_dim

        self.WNN = WNN(manifold_dim)
        self.GradHNN = GradHNN(manifold_dim)
    
    def forward(self, u):
        W_u = self.WNN(u)
        grad_H_u = self.GradHNN(u)

        #X_H is such that W_u^T * X_H = grad_H
        X_H_u = torch.linalg.solve(W_u.permute(0, 2, 1), grad_H_u)
        return X_H_u
    
    def predict(self, u):
        self.eval()
        with torch.no_grad():
            W_u = self.WNN(u.unsqueeze(0))
            grad_H_u = self.GradHNN(u.unsqueeze(0))

            #X_H is such that W_u^T * X_H = grad_H
            X_H_u = torch.linalg.solve(W_u.permute(0, 2, 1), grad_H_u)
        return X_H_u.squeeze(0)

    def get_closeness_loss(self, u):
        return self.WNN.get_closeness_loss(u)

class WNN(nn.Module):
    """
    The model that estimates the gradient of the hamiltonian.
    """

    def __init__(self, manifold_dim):
        super().__init__()
        self.manifold_dim = manifold_dim

        self.WNN = nn.Sequential(
            nn.Linear(self.manifold_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, (self.manifold_dim * (self.manifold_dim - 1)) // 2)
        )
    
    def forward(self, u):
        W_u_array = self.WNN(u)
        row_idx, col_idx = torch.triu_indices(self.manifold_dim, self.manifold_dim, offset=1)
        W_u = torch.zeros((u.shape[0], self.manifold_dim, self.manifold_dim), device=u.device)
        W_u[:, row_idx, col_idx] = W_u_array
        W_u[:, col_idx, row_idx] = -W_u_array
        return W_u
    
    def get_closeness_loss(self, u):
        #get the gradient of w_ij with respect to u for all i,j
        JW_u_array = vmap(jacrev(self.WNN))(u)
        row_idx, col_idx = torch.triu_indices(self.manifold_dim, self.manifold_dim, offset=1)
        JW_u = torch.zeros((u.shape[0], self.manifold_dim, self.manifold_dim, self.manifold_dim), device=u.device)
        JW_u[:, row_idx, col_idx, :] = JW_u_array
        JW_u[:, col_idx, row_idx, :] = -JW_u_array

        JW_u1 = JW_u.permute(0, 3, 1, 2)
        JW_u2 = JW_u.permute(0, 2, 3, 1)

        S = JW_u + JW_u1 + JW_u2

        return (torch.sqrt(torch.sum(S**2, dim=(1, 2, 3))) / (self.manifold_dim)**3).mean()


class GradHNN(nn.Module):
    """
    The model that estimates the gradient of the hamiltonian.
    """

    def __init__(self, manifold_dim):
        super().__init__()
        self.manifold_dim = manifold_dim

        self.HNN = nn.Sequential(
            nn.Linear(self.manifold_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.manifold_dim)
        )
    
    def forward(self, u):
        grad_H_u = self.HNN(u)
        return grad_H_u
    
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sanity check the model
    manifold_dim = 4
    model = ClosenessEnforcer(manifold_dim).to(device)
    u = torch.randn((10, manifold_dim), device=device)
    l = model.get_closeness_loss(u)
    print(l)