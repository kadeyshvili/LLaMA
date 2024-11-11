import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_heads, emb_dim, max_batch_size, max_seq_len, device):
        super().__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size
        self.head_dim = self.emb_dim // self.n_heads
        self.max_seq_len = max_seq_len
        self.queries = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.keys = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.values = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.device = device


    def forward(self, x, rope, mask):
        batch_size, sequence_len, _ = x.shape
        x = x.to(self.device)
        queries = self.queries(x).view(batch_size, sequence_len, self.n_heads, self.head_dim)
        keys = self.keys(x).view(batch_size, sequence_len, self.n_heads, self.head_dim)
        values = self.values(x).view(batch_size, sequence_len, self.n_heads, self.head_dim)

        queries, keys = rope(queries, keys)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_len, -1)

        return self.out(output)