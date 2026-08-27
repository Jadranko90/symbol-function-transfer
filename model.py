"""Small GPT: token + position embeddings, stacked transformer blocks, LM head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Block


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=64,
                 n_head=4, n_layer=3, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        emb = (self.token_embedding(idx)
               + self.position_embedding(torch.arange(T, device=idx.device)))
        h = self.ln_f(self.blocks(emb))
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T),
                                   ignore_index=-100)
        return logits, loss


if __name__ == "__main__":
    from batching import Batcher
    b = Batcher("corpus.txt")
    model = GPT(vocab_size=b.tok.vocab_size, block_size=b.max_len - 1)
    print(f"vocab {b.tok.vocab_size}  block {b.max_len - 1}  "
          f"params {sum(p.numel() for p in model.parameters()):,}")
    x, y = b.get_batch(4)
    logits, loss = model(x, y)
    print(f"logits {tuple(logits.shape)}  loss {loss.item():.4f} "
          f"(random init ~ln(vocab) = "
          f"{torch.log(torch.tensor(float(b.tok.vocab_size))).item():.4f})")
