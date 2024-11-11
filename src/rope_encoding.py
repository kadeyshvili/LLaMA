import torch
import torch.nn as nn 


class RoPE(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device, theta = 10000.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        freqs = 1.0 / (theta ** (torch.arange(0, emb_dim, 2).float() / emb_dim)).to(self.device)
        t = torch.arange(max_seq_len, device = self.device)
        freqs = torch.outer(t, freqs)
        self.theta = torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, queries, keys):
        batch_size, seq_len, n_head, head_dim = queries.shape
        queries_complex = torch.view_as_complex(queries.reshape(batch_size, seq_len, n_head, -1, 2))
        keys_complex = torch.view_as_complex(keys.reshape(batch_size, seq_len, n_head, -1, 2))
        theta = self.theta.unsqueeze(0).unsqueeze(2)
        queries_roated = queries_complex * theta
        keys_roated = keys_complex * theta
        result_queries = torch.view_as_real(queries_roated).reshape(batch_size, seq_len, n_head, head_dim)
        result_keys = torch.view_as_real(keys_roated).reshape(batch_size, seq_len, n_head, head_dim)
        return result_queries, result_keys