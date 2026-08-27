"""Training loop with a held-out validation split.

The split matters for this experiment: training loss falling only shows the model
fit the units it saw. The claim being tested is about generalisation to material
the model was never trained on, so the best-validation weights are kept and the
final (overfit) weights are discarded.
"""

import time
import torch
from batching import Batcher, IGNORE_INDEX
from model import GPT

STEPS = 2000
BATCH_SIZE = 32
LR = 1e-3
EVAL_EVERY = 500
EVAL_ITERS = 20
VAL_FRAC = 0.1
SEED = 0
CKPT = "model.pt"


def main():
    torch.manual_seed(SEED)
    b = Batcher("corpus.txt")
    block_size = b.max_len - 1

    n = len(b.encoded)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    n_val = int(n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"units: {n}  ->  train {len(train_idx)}  val {len(val_idx)}")

    def get_split_batch(split):
        pool = train_idx if split == "train" else val_idx
        pick = torch.randint(len(pool), (BATCH_SIZE,)).tolist()
        L = b.max_len - 1
        xs, ys = [], []
        for p in pick:
            unit = b.encoded[pool[p]]
            xs.append(b._pad(unit[:-1], L, b.pad_id))
            ys.append(b._pad(unit[1:], L, IGNORE_INDEX))
        return torch.tensor(xs), torch.tensor(ys)

    model = GPT(vocab_size=b.tok.vocab_size, block_size=block_size)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}   "
          f"block_size: {block_size}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(EVAL_ITERS)
            for i in range(EVAL_ITERS):
                x, y = get_split_batch(split)
                _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    print(f"\n{'step':>6s}  {'train':>7s}  {'val':>7s}   {'elapsed':>8s}")
    t0 = time.time()
    best_val, best_step = float("inf"), -1
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            L = estimate_loss()
            flag = ""
            if L["val"] < best_val:
                best_val, best_step = L["val"], step
                torch.save({"model_state": model.state_dict(),
                            "vocab_size": b.tok.vocab_size,
                            "block_size": block_size,
                            "n_embed": model.n_embed,
                            "stoi": b.tok.stoi,
                            "step": step,
                            "val_loss": L["val"]}, CKPT)
                flag = "  <- best, saved"
            print(f"{step:6d}  {L['train']:7.4f}  {L['val']:7.4f}   "
                  f"{time.time()-t0:7.1f}s{flag}")

        x, y = get_split_batch("train")
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f"\nbest val {best_val:.4f} at step {best_step}  ->  {CKPT}")


if __name__ == "__main__":
    main()
