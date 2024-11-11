import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

