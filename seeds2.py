"""Study 2 replication across independent runs.

UNIT OF ANALYSIS
  Each seed is a fresh corpus, a fresh initialisation and a fresh training run --
  genuinely independent replications. So SEED is the unit: each contributes one
  slope, one r, one contrast, and those go into one-sample t-tests.

  Pooling the ~600 generations per seed and regressing would give a far smaller
  p-value and would be wrong. Generations from the same symbol on the same model
  are many samples of ONE thing, not many independent observations.
"""

import io, contextlib, random
import torch
from scipy import stats

import corpus2
import train as train_mod
from tokenizer import Tokenizer, load_corpus_file
import measure2

N_SEEDS = 6
STEPS = 1500
REPS = 20
measure2.START_DISTS = [2, 3, 4]


def run_seed(seed):
    corpus2.save_corpus(corpus2.build_corpus(seed=seed), "corpus.txt")

    train_mod.STEPS = STEPS
    train_mod.EVAL_EVERY = STEPS
    train_mod.SEED = seed
    with contextlib.redirect_stdout(io.StringIO()):
        train_mod.main()

    tok = Tokenizer(load_corpus_file("corpus.txt"))
    model, ck = measure2.load()
    bs = ck["block_size"]
    random.seed(seed * 31)
    torch.manual_seed(seed * 31)

    P = lambda s: measure2.probe(model, tok, bs, s, reps=REPS)[0]
    out = {"STIM_A": P("STIM_A"), "foo": P("foo"), "bar": P("bar"), "ctrl": P("ctrl")}
    qs = [P(f"q{i}") for i in range(1, 7)]
    out["q"] = qs

    X, Y = torch.tensor([1., 2., 3., 4., 5., 6.]), torch.tensor(qs)
    out["slope"] = (((X - X.mean()) * (Y - Y.mean())).sum()
                    / ((X - X.mean()) ** 2).sum()).item()
    out["r"] = float(torch.corrcoef(torch.stack([X, Y]))[0, 1])
    return out


def report(name, vals):
    t, p = stats.ttest_1samp(vals, 0.0, alternative="greater")
    v = torch.tensor(vals)
    per = " ".join(f"{x:+.2f}" for x in vals)
    print(f"  {name:<26s} {v.mean():7.2f} ± {v.std():5.2f}   "
          f"t={t:6.2f}  p={p:.4f}  d={(v.mean()/v.std()):5.2f}   {per}")


def main():
    rows = [run_seed(s) for s in range(N_SEEDS) if not print(f"seed {s} ...", flush=True)]
    g = lambda k: [r[k] for r in rows]

    print("\n" + "=" * 92)
    print(f"STUDY 2 — {N_SEEDS} independent replications (seed = unit of analysis)")
    print("=" * 92)

    v = torch.tensor(g("STIM_A"))
    print(f"\nGATE — anchor is trained on movement, announces 2")
    print(f"  STIM_A moved {v.mean():.2f} ± {v.std():.2f}  (want ~2.0)")

    print("\nTRANSFER — none of these ever appear in a movement unit")
    print(f"  {'test':<26s} {'mean':>7s}   {'sd':>5s}   "
          f"{'t':>6s}  {'p':>7s}  {'d':>5s}   per-seed")
    report("slope (announced->moved)", g("slope"))
    report("foo - ctrl", [a - b for a, b in zip(g("foo"), g("ctrl"))])
    report("bar - ctrl", [a - b for a, b in zip(g("bar"), g("ctrl"))])
    report("bar - foo  (ordering)", [a - b for a, b in zip(g("bar"), g("foo"))])
    report("r (announced vs moved)", g("r"))

    print("\nDOSE-RESPONSE — mean movement per announced intensity")
    qs = torch.tensor([r["q"] for r in rows])
    print(f"  {'announced':>10s} {'moved':>8s} {'sd':>6s}")
    for i in range(6):
        c = qs[:, i]
        print(f"  {i+1:10d} {c.mean():8.2f} {c.std():6.2f}")

    print("\nRAW MEANS")
    for k in ("STIM_A", "foo", "bar", "ctrl"):
        v = torch.tensor(g(k))
        print(f"  {k:8s} {v.mean():6.2f} ± {v.std():.2f}")
    print(f"\nOne-tailed, n = {N_SEEDS} seeds. Low-powered: a non-significant p "
          f"here is not evidence of absence.")


if __name__ == "__main__":
    main()
