import torch
from torch import nn
import torch.nn.functional as F

from src.rope_encoding import RoPE
from src.rms_norm import RMSNorm
from src.attention import Attention
from src.feed_forward import FeedForward



class TransformerBlock(nn.Module):
    def __init__(self, n_heads, emb_dim, max_batch_size, max_seq_len, norm_eps, device):
        super().__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // self.n_heads
        self.attention = Attention(n_heads, emb_dim, max_batch_size, max_seq_len, device)
        self.pre_norm = RMSNorm(self.emb_dim, eps=norm_eps)
        self.device = device
        self.feed_forward = FeedForward(emb_dim=self.emb_dim, hidden_dim=self.emb_dim * 4, device=self.device)

        self.post_norm = RMSNorm(self.emb_dim, eps=norm_eps)



    def forward(self, x, rope, mask):
        h = x + self.attention.forward(self.pre_norm(x), rope, mask)
        out = h + self.feed_forward.forward(self.post_norm(h))
        return out


class LLaMA(nn.Module):
    def __init__(self, n_heads, emb_dim, vocab_size, n_layers, norm_eps, max_seq_len, max_batch_size, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)

        self.rope = RoPE(emb_dim // n_heads, max_seq_len * 2, device)
        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(n_heads, emb_dim, max_batch_size, max_seq_len, norm_eps, device=device))
        self.norm = RMSNorm(emb_dim, eps=norm_eps)
        self.output = nn.Linear(emb_dim, vocab_size, bias=False)
        self.device = device


    def forward(self, input_ids):
        sequence_len = input_ids.shape[1]
        h = self.embedding(input_ids)
        self.rope.theta = self.rope.theta[ : sequence_len].to(h.device)
        mask = None
        if sequence_len > 1:
            mask = torch.full((1, 1, sequence_len, sequence_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = h.to(self.device)
            h = layer(h, self.rope, mask)
        h = self.norm(h).to(self.device)
        output = self.output(h)
        return output
