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

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.means.shape

        with torch.no_grad():
            return torch.normal(self.means.expand(shape), 1) / self.precisions.expand(
                shape
            )


class BivariateNormal:
    def __init__(self, loc, covariance_matrix):
        # TODO: should check that things are reasonable, but I'm lazy
        self.means = loc

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
        z = (
            (deltas.unsqueeze(1) @ self.precision_matrices @ deltas.unsqueeze(2))
            .squeeze(2)
            .squeeze(1)
        )
        return -0.5 * (z + torch.log(self.determinants)) - LOG_2PI

    # TODO: implement sample() allowing for singular covariance


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

    # TODO: implement sample() allowing for singular covariance
