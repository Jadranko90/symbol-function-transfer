"""Study 2 measurement.

Probe:  AG ## SYM  ->  model continues  a03 AG ##### SYM
DV = (distance after) - (distance before), i.e. how far the agent fled.

The probe carries no announcements and no relational statement -- only the symbol
itself. So whatever magnitude appears has to come from the symbol's learned
representation, not from anything in the context window.
"""

import torch
from tokenizer import Tokenizer, load_corpus_file
from model import GPT

MARK, AG = "#", "AG"
TEMP, MAX_NEW = 0.8, 20
START_DISTS = [2, 3, 4, 5]


def load(ckpt="model.pt"):
    ck = torch.load(ckpt)
    m = GPT(vocab_size=ck["vocab_size"], block_size=ck["block_size"],
            n_embed=ck["n_embed"])
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck


@torch.no_grad()
def gen(model, ids, bs):
    for _ in range(MAX_NEW):
        logits, _ = model(ids[:, -bs:])
        p = torch.softmax(logits[:, -1, :] / TEMP, dim=-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    return ids


def next_dist(toks):
    if AG not in toks:
        return None
    k = toks.index(AG)
    n, j = 0, k + 1
    while j < len(toks) and toks[j] == MARK:
        n += 1
        j += 1
    return n if j < len(toks) else None


def probe(model, tok, bs, sym, reps=40):
    out = []
    for d in START_DISTS:
        prompt = [AG] + [MARK] * d + [sym]
        base = torch.tensor([tok.encode(prompt)])
        for _ in range(reps):
            g = gen(model, base.clone(), bs)
            nd = next_dist(tok.decode(g[0].tolist()[len(prompt):]))
            if nd is not None:
                out.append(nd - d)
    if not out:
        return (float("nan"),) * 3
    t = torch.tensor(out, dtype=torch.float)
    return t.mean().item(), t.std().item(), len(out)


def main():
    tok = Tokenizer(load_corpus_file("corpus.txt"))
    model, ck = load()
    bs = ck["block_size"]
    print(f"model.pt  val {ck.get('val_loss', float('nan')):.4f}  block {bs}\n")

    rows = [("STIM_A  anchor, announces 2", "STIM_A", 2),
            ("foo     announces 4  HELD OUT", "foo", 4),
            ("bar     announces 6  HELD OUT", "bar", 6),
            ("ctrl    no information", "ctrl", None)]
    print(f"{'condition':<32s} {'moved':>7s} {'sd':>6s} {'n':>5s}  announced")
    for label, sym, ann in rows:
        m, sd, n = probe(model, tok, bs, sym)
        print(f"{label:<32s} {m:7.2f} {sd:6.2f} {n:5d}  {ann if ann else '-'}")

    print("\n--- DOSE-RESPONSE (all held out of movement) ---")
    print(f"{'symbol':<8s} {'announced':>10s} {'moved':>8s} {'sd':>6s}")
    xs, ys = [], []
    for i in range(1, 7):
        m, sd, _ = probe(model, tok, bs, f"q{i}")
        print(f"{'q'+str(i):<8s} {i:10d} {m:8.2f} {sd:6.2f}")
        xs.append(float(i))
        ys.append(m)
    X, Y = torch.tensor(xs), torch.tensor(ys)
    r = float(torch.corrcoef(torch.stack([X, Y]))[0, 1])
    slope = (((X - X.mean()) * (Y - Y.mean())).sum()
             / ((X - X.mean()) ** 2).sum()).item()
    print(f"\n  r = {r:.3f}   slope = {slope:.2f}  (1.0 = full magnitude transfer)")


if __name__ == "__main__":
    main()
