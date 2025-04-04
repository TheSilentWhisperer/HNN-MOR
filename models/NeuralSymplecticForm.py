import torch
import torch.nn as nn
from torch.func import jacrev, vmap

class NeuralSymplecticForm(nn.Module):
    
    """
    The model that estimates the hamiltonian vector field.
    """

    def __init__(self, manifold_dim):
        super().__init__()
        
        # A model that estimates the symplectic form
        self.W = WNN(manifold_dim)
        self.grad_H = GradHNN(manifold_dim)

        
    def forward(self, u):
        W_u = self.W(u)
        grad_H_u = self.grad_H(u)

        #X_H is such that W_u^T * X_H = grad_H
        X_H_u = torch.linalg.solve(W_u.permute(0, 2, 1), grad_H_u)
        return X_H_u
    
    def predict(self, u):
        self.eval()
        with torch.no_grad():
            W_u = self.W(u.unsqueeze(0))
            grad_H_u = self.grad_H(u.unsqueeze(0))

            #X_H is such that W_u^T * X_H = grad_H
            X_H_u = torch.linalg.solve(W_u.permute(0, 2, 1), grad_H_u)
        return X_H_u.squeeze(0)
    
class WNN(nn.Module):
    """
    The model that estimates the symplectic form.
    """

    def __init__(self, manifold_dim):
        super().__init__()
        self.manifold_dim = manifold_dim

        # The underlying network that estimates the 1-form
        self.fNN = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.manifold_dim)
        )

    def forward(self, u):
        #get the jacobian of fNN with respect to u
        Jf_u = vmap(jacrev(self.fNN))(u)
        Jf_uT = Jf_u.permute(0, 2, 1)

        #get the symplectic form
        W_u = Jf_uT - Jf_u
        return W_u
    
class GradHNN(nn.Module):
    """
    The model that estimates the gradient of the hamiltonian.
    """

    def __init__(self, manifold_dim):
        super().__init__()
        self.manifold_dim = manifold_dim

        self.HNN = nn.Sequential(
            nn.Linear(4, 128),
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

    #Sanity check the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifold_dim = 4

    from utils.data.DynDataset import DynDataset

    dataset = DynDataset("datasets/2d_pend/unact_small_dq0.pickle", device, data_type="point")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    q, dq, ddq = next(iter(dataloader))
    
    print("q", q.shape)
    print("dq", dq.shape)
    print("ddq", ddq.shape)
    u = torch.cat((q, dq), dim=1)
    print("u", u.shape)

    model = NeuralSymplecticForm(manifold_dim).to(device)
    x_H_u = model(u)
    print("x_H_u", x_H_u.shape)