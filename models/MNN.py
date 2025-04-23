import torch
from torch import nn
from torch.func import vmap, jacrev
from models.DifferentiableLayers import DifferentialLinear, DifferentialSoftplus

def vect_to_mat(v, n):
    """
    Converts a batch of vectors v of shape (batch_size, n*(n+1)//2)
    into symmetric matrices of shape (batch_size, n, n),
    filling the lower-triangular part and mirroring it.
    """

    col_idx, row_idx = torch.tril_indices(n, n, device=v.device)

    is_batched = v.ndim == 2
    if is_batched:
        batch_size = v.shape[0]
        mat = torch.zeros(batch_size, n, n, device=v.device)
        mat[:, col_idx, row_idx] = v
        mat[:, row_idx, col_idx] = v
    else:
        mat = torch.zeros(n, n, device=v.device)
        mat = torch.index_put(mat, (col_idx, row_idx), v)
        mat = torch.index_put(mat, (row_idx, col_idx), v)
    
    return mat

class MNN(nn.Module):
    """
    Takes a point (q) in the manifold, maps it to a tangent vector U(q) using a neural network,
    and computes the mass-inertia matrix M(q) using the exponential map.
    """
    def __init__(self, n_manifold):
        self.n_manifold = n_manifold
        super().__init__()

        self.UvectNN = nn.Sequential(
            DifferentialLinear(n_manifold, 128),
            DifferentialSoftplus(),
            DifferentialLinear(128, 128),
            DifferentialSoftplus(),
            DifferentialLinear(128, n_manifold*(n_manifold+1)//2)
        )

        #basepoint in the manifold from which we compute the exponential map
        #P shape : (n_manifold, n_manifold)
        self.P = torch.zeros(n_manifold * (n_manifold + 1) // 2)

    def mat_exp(self, U):
        """
        Computes the matrix exponential of a symmetric matrix U.
        """
        eigvals, eigvecs = torch.linalg.eigh(U)
        exp_eigvals = torch.exp(eigvals)
        exp_U = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-1, -2)
        return exp_U

    def get_M_from_Uvect(self, Uvect, P):
        U = vect_to_mat(Uvect, self.n_manifold)
        return self.mat_exp(U)

    def forward(self, input):
        """
        q shape: (batch_size, n_manifold)
        dq shape: (batch_size, n_manifold)
        """
        q, dqdq = input["x"], input["dxdq"]
        is_batched = q.ndim == 2

        #compute Uvect, dUvect
        output = self.UvectNN({"x": q, "dxdq": dqdq})
        Uvect, dUvectdq = output["x"], output["dxdq"]
        Uvect = Uvect.clone().requires_grad_()

        #compute M, dMdq
        P = self.P.unsqueeze(0).repeat(q.shape[0], 1)
        M = self.get_M_from_Uvect(Uvect, P)

        if not is_batched:
            return M, None

        # dMdUvect = batch_jacobian(M, Uvect)
        dMdUvect = vmap(jacrev(self.get_M_from_Uvect, 0))(Uvect, P)
        dMdq = torch.einsum('bijk,bkl->bijl', dMdUvect, dUvectdq)
        return {"x": M, "dxdq": dMdq}