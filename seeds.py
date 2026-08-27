"""
Multi-seed replication for the RFT-LLM experiment.

One seed is an anecdote. This runs the WHOLE pipeline independently N times --
fresh corpus, fresh weight initialisation, fresh training -- and reports each
measure across seeds, so we can see which findings are stable and which were
luck.

Per seed:
  build corpus -> train -> run the probe battery

The battery, in the order the argument needs them:
  1. ANCHORS          -- did training work at all? (gate: everything else is
                         meaningless if the grounded stimuli don't behave)
  2. MANIPULATION     -- were '=' and '!=' learned AS OPERATIONS, in the exact
                         format they were trained on? (gate: you cannot test
                         transfer through a relation the model never acquired)
  3. IN-CONTEXT       -- with the statement in the window, does a novel symbol
                         acquire the function? does it invert under '!='? does
                         it track WHICH anchor?
  4. PRESENCE CONTROL -- is it the relation, or just the anchor token sitting
                         in the context window? (the alternative explanation)
  5. BARE             -- with the statement REMOVED, does anything persist in
                         the symbol's own representation?
"""

import io, random, contextlib
import torch

import corpus_min
import train as train_mod
from tokenizer import Tokenizer, load_corpus_file
from measure import load_model, generate, parse_next_distance
from corpus_min import _snapshot_pair, START_RANGE, EQ, NEQ

N_SEEDS = 5
STEPS   = 1500
REPS    = 50
H = ["h00", "h01", "h02", "h03", "h04"]
K = ["k00", "k01", "k02", "k03", "k04"]


def probe(model, tok, bs, prefix, sym, reps=REPS):
    """prefix tokens (may contain '{d}' placeholder) + AG dNN sym -> mean delta"""
    out = []
    for _ in range(reps):
        d = random.randint(*START_RANGE)
        pre = [f"d{d:02d}" if t == "{d}" else t for t in prefix]
        prompt = pre + ["AG", f"d{d:02d}", sym]
        base = torch.tensor([tok.encode(prompt)])
        g = generate(model, base.clone(), 4, 0.8, bs)
        nd = parse_next_distance(tok.decode(g[0].tolist()[len(prompt):]))
        if nd is not None:
            out.append(nd - d)
    return sum(out) / len(out) if out else float("nan")


def manipulation(model, tok, bs, reps=REPS):
    """'=' effect and '!=' effect in the exact trained format."""
    def cell(rel, dirA):
        out = []
        for _ in range(reps):
            A, B = random.sample([f"r{i:03d}" for i in range(120)], 2)
            partA, _ = _snapshot_pair(A, dirA, random.randint(*START_RANGE))
            d = random.randint(*START_RANGE)
            prompt = [A, rel, B, "|"] + partA + ["|"] + ["AG", f"d{d:02d}", B]
            base = torch.tensor([tok.encode(prompt)])
            g = generate(model, base.clone(), 4, 0.8, bs)
            nd = parse_next_distance(tok.decode(g[0].tolist()[len(prompt):]))
            if nd is not None:
                out.append(nd - d)
        return sum(out) / len(out) if out else float("nan")
    eq  = cell(EQ, "away")  - cell(EQ, "toward")
    neq = cell(NEQ, "away") - cell(NEQ, "toward")
    return eq, neq


def run_seed(seed):
    # ---- fresh corpus -------------------------------------------------
    corpus, _ = corpus_min.build_corpus(seed=seed)
    corpus_min.save_corpus(corpus, "corpus.txt")

    # ---- fresh training (silenced) ------------------------------------
    train_mod.STEPS = STEPS
    train_mod.EVAL_EVERY = STEPS  # only evaluate at start and end
    train_mod.SEED = seed
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            train_mod.main()
        except KeyError:
            pass          # train.py's demo prompt still uses the old '#' tokens
    val = float([l for l in buf.getvalue().splitlines() if "best val" in l][0]
                .split()[2])

    # ---- probes -------------------------------------------------------
    tok = Tokenizer(load_corpus_file("corpus.txt"))
    with contextlib.redirect_stdout(io.StringIO()):
        model, ck = load_model()
    bs = ck["block_size"]
    random.seed(seed * 977); torch.manual_seed(seed * 977)

    r = {"val": val}
    r["STIM_A"] = probe(model, tok, bs, [], "STIM_A")
    r["STIM_T"] = probe(model, tok, bs, [], "STIM_T")
    r["eq_eff"], r["neq_eff"] = manipulation(model, tok, bs)
    r["foo_eq_A"]  = probe(model, tok, bs, ["foo", "=",  "STIM_A", "|"], "foo")
    r["foo_neq_A"] = probe(model, tok, bs, ["foo", "!=", "STIM_A", "|"], "foo")
    r["foo_eq_T"]  = probe(model, tok, bs, ["foo", "=",  "STIM_T", "|"], "foo")
    r["h_ctx"] = sum(probe(model, tok, bs, [s, "=",  "STIM_A", "|"], s) for s in H) / len(H)
    r["k_ctx"] = sum(probe(model, tok, bs, [s, "!=", "STIM_A", "|"], s) for s in K) / len(K)
    r["presence_A"] = probe(model, tok, bs, ["AG", "{d}", "STIM_A", "|"], "foo")
    r["foo_bare"] = probe(model, tok, bs, [], "foo")
    r["ctrl_bare"] = probe(model, tok, bs, [], "ctrl")
    r["h_bare"] = sum(probe(model, tok, bs, [], s) for s in H) / len(H)
    r["k_bare"] = sum(probe(model, tok, bs, [], s) for s in K) / len(K)
    return r


def main():
    rows = []
    for s in range(N_SEEDS):
        print(f"running seed {s} ...", flush=True)
        rows.append(run_seed(s))

    def col(k):
        v = [r[k] for r in rows]
        t = torch.tensor(v)
        return t.mean().item(), t.std().item(), v

    print("\n" + "=" * 74)
    print(f"{N_SEEDS} SEEDS -- fresh corpus, fresh init, fresh training each")
    print("=" * 74)
    groups = [
        ("1. ANCHORS (gate: did training work)", [
            ("STIM_A  grounded away", "STIM_A", "> 0"),
            ("STIM_T  grounded toward", "STIM_T", "< 0")]),
        ("2. MANIPULATION (gate: were the operators learned)", [
            ("'='  effect", "eq_eff", "strongly > 0"),
            ("'!=' effect", "neq_eff", "strongly < 0")]),
        ("3. IN-CONTEXT derivation (statement in window)", [
            ("foo =  STIM_A", "foo_eq_A", "> 0"),
            ("foo != STIM_A  (inverts?)", "foo_neq_A", "< 0"),
            ("foo =  STIM_T  (tracks anchor?)", "foo_eq_T", "< 0"),
            ("held-out h.. =  STIM_A", "h_ctx", "> 0"),
            ("held-out k.. != STIM_A", "k_ctx", "< 0")]),
        ("4. PRESENCE CONTROL (relation, or just the anchor token?)", [
            ("STIM_A present, NO relation", "presence_A", "~ 0")]),
        ("5. BARE (statement removed -- anything durable?)", [
            ("foo", "foo_bare", "> 0 if durable"),
            ("ctrl  (unlinked baseline)", "ctrl_bare", "~ 0"),
            ("held-out h.. group", "h_bare", "> 0 if durable"),
            ("held-out k.. group", "k_bare", "< 0 if durable")]),
    ]
    for title, items in groups:
        print(f"\n{title}")
        print(f"  {'measure':<34s} {'mean':>7s} {'sd':>6s}   {'predicted':<16s} per-seed")
        for label, key, pred in items:
            m, sd, v = col(key)
            per = " ".join(f"{x:+.1f}" for x in v)
            print(f"  {label:<34s} {m:7.2f} {sd:6.2f}   {pred:<16s} {per}")
    m, sd, _ = col("val")
    print(f"\nval loss: {m:.3f} (sd {sd:.3f})")


if __name__ == "__main__":
    main()
