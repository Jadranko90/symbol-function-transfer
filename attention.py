"""Causal self-attention: one head, multi-head, feed-forward, and a full block."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return (out, weights) if return_weights else out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert n_embed % num_heads == 0
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
