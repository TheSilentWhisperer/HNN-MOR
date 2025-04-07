import numpy as np

def PCA(X):
    # we need the biggest eigenvalues of X.T @ X
    # we can use the SVD to get them
    U, S, V = np.linalg.svd(X)
    # the biggest eigenvalues are the first two singular values
    return V[:2, :].T