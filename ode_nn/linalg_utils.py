import numpy as np
import torch

# used by MVN pdfs
LOG_2PI = np.log(2 * np.pi)


class UnivariateNormal:
    def __init__(self, loc, covariance_matrix):
        self.means = loc.squeeze(1)
        self.precisions = 1 / covariance_matrix.squeeze(2).squeeze(1)

    def log_prob(self, value):
        value = value.squeeze(1)
        zsq = (value - self.means) ** 2 * self.precisions
        return -0.5 * (zsq - torch.log(self.precisions) + LOG_2PI)

    def sample(self, n=1):
        with torch.no_grad():
            zs = torch.randn(n, *self.means.shape)
            ret = zs / self.precisions.sqrt().unsqueeze(0) + self.means.unsqueeze(0)
            return ret.unsqueeze(-1)
            # return shape [n, *means.shape, 1] for consistency with Bi/Trivariate


class BivariateNormal:
    def __init__(self, loc, covariance_matrix):
        # TODO: should check that things are reasonable, but I'm lazy
        self.means = loc
        self.covariance_matrices = covariance_matrix

        a = covariance_matrix[:, 0, 0]
        b = covariance_matrix[:, 0, 1]
        d = covariance_matrix[:, 1, 1]
        self.determinants = a * d - b * b
        mb = -b
        self.precision_matrices = (
            torch.stack([d, mb, mb, a], 1).reshape(a.shape[0], 2, 2)
            / self.determinants[:, None, None]
        )

    def log_prob(self, value):
        deltas = value - self.means
        z = deltas.unsqueeze(1) @ self.precision_matrices @ deltas.unsqueeze(2)
        return -0.5 * (z.squeeze(2).squeeze(1) + self.determinants.log()) - LOG_2PI

    def get_choleskies(self):
        # TODO: would be good to cache this. theoretically, though,
        #       we might want to call it either with gradients (for some reason)
        #       or without gradients (for sampling), and plain caching wouldn't
        #       handle this global state; since we only call it more than once
        #       when we do rejection sampling, and it's not that big a deal,
        #       not bothering for now.
        a = self.covariance_matrices[:, 0, 0]
        b = self.covariance_matrices[:, 0, 1]
        d = self.covariance_matrices[:, 1, 1]
        zeros = torch.zeros_like(a)
        sqrt_a = a.sqrt()
        return torch.stack(
            [sqrt_a, b / sqrt_a, zeros, (d - b.square() / a).sqrt()], 1
        ).reshape(a.shape[0], 2, 2)

    def sample(self, n=1):
        with torch.no_grad():
            zs = torch.randn(n, *self.means.shape)
            Us = self.get_choleskies()
            U_zs = Us.transpose(2, 1).unsqueeze(0) @ zs.unsqueeze(-1)
            return U_zs.squeeze(-1) + self.means.unsqueeze(0)


class TrivariateNormal:
    def __init__(self, loc, covariance_matrix):
        # TODO: should check that things are reasonable, but I'm lazy
        self.means = loc

        a = covariance_matrix[:, 0, 0]
        b = covariance_matrix[:, 0, 1]
        c = covariance_matrix[:, 0, 2]
        d = covariance_matrix[:, 1, 1]
        e = covariance_matrix[:, 1, 2]
        f = covariance_matrix[:, 2, 2]

        A = d * f - e * e
        B = c * e - b * f
        C = b * e - c * d
        D = a * f - c * c
        E = b * c - a * e
        F = a * d - b * b

        self.determinants = a * A + b * B + c * C

        invs = torch.stack([A, B, C, B, D, E, C, E, F], 1).reshape(-1, 3, 3)
        self.precision_matrices = invs / self.determinants[:, None, None]

    def log_prob(self, value):
        deltas = value - self.means
        z = (
            (deltas.unsqueeze(1) @ self.precision_matrices @ deltas.unsqueeze(2))
            .squeeze(2)
            .squeeze(1)
        )
        return -0.5 * (z + torch.log(self.determinants) + 3 * LOG_2PI)

    # TODO: implement sample()
