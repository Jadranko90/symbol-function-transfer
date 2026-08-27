"""Batching.

Each UNIT is a separate example, padded to a common length and stacked as its own
row. This differs from the usual random-window sampling, which would let one unit
bleed into the next -- here a relation must never leak into a behaviour episode.
Padded target positions are set to IGNORE_INDEX so they contribute no loss.
"""

import torch
import random
from tokenizer import Tokenizer, load_corpus_file

IGNORE_INDEX = -100


class Batcher:
    def __init__(self, corpus_path="corpus.txt"):
        self.corpus = load_corpus_file(corpus_path)
        self.tok = Tokenizer(self.corpus)
        self.pad_id = self.tok.pad_id
        self.encoded = [self.tok.encode(u) for u in self.corpus]
        self.max_len = max(len(u) for u in self.encoded)

    def _pad(self, ids, length, value):
        return ids + [value] * (length - len(ids))

    def get_batch(self, batch_size):
        """(x, y) of shape (batch_size, max_len - 1); y is x shifted by one."""
        idx = random.sample(range(len(self.encoded)), batch_size)
        L = self.max_len - 1
        x_rows, y_rows = [], []
        for i in idx:
            unit = self.encoded[i]
            x_rows.append(self._pad(unit[:-1], L, self.pad_id))
            y_rows.append(self._pad(unit[1:], L, IGNORE_INDEX))
        return torch.tensor(x_rows), torch.tensor(y_rows)


if __name__ == "__main__":
    b = Batcher("corpus.txt")
    x, y = b.get_batch(3)
    print(f"vocab {b.tok.vocab_size}  units {len(b.encoded)}  max_len {b.max_len}")
    print(f"x {tuple(x.shape)}  y {tuple(y.shape)}")
