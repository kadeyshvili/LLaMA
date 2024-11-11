import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, bias=False)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, bias=False)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        swish = F.silu(self.w1(x))
        return self.w2(swish * self.w3(x))