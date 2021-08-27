import numpy as np
import torch

"""
The regularizer we want to use is based on a kernel K = Z Z^T
    min_{w, delta} ||w||^2 + lambda ||delta||^2   s.t. theta = Z w + delta
                (RKHS norm of Z w)  (departure)
    = min_w ||w||^2 + lambda ||theta - Z w||^2
    = theta^T ( lambda (I - K inv(K + 1/lambda I)) ) theta
    = theta^T reg theta
where reg = lambda (I - K inv(K + 1/lambda I))
will have eigenvectors the same as K (the left singular vectors of Z)
and eigenvalues 1 / (s_i + 1/lambda), where s_i are the eigenvalues of K
                                     (the squared singular values of Z)

(To derive this, just expand the norms and set the derivative to zero.
Left as an exercise to Aram.)
"""

def get_regularizer(embeddings: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Returns a matrix R such that the regularizer above is ||R theta||^2.
    """
    U, S, Vh = torch.linalg.svd(embeddings)
    eigs = 1 / (S**2 + 1 / lam)
    return torch.sqrt(eigs)[:, np.newaxis] * U.t()
