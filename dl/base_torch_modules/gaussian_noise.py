from torch import nn
import torch as t


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            # random sign matrix, shape of x only -1 or 1
            sign = t.rand_like(x)
            sign = 2 * (sign > 0.5) - 1
            # x has mean of 0.5 with variance 0.5, add noise according to sigma
            return x + sign * t.rand_like(x) * self.sigma
        else:
            return x
