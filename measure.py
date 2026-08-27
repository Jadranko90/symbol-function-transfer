"""
Measurement for the RFT-LLM experiment.

This is the file that tests the hypothesis. Everything before it built the
subject; this one runs the behavioural probe.

THE PROBE
  Present the agent at a distance from a SYMBOL, and let the model continue:

      prompt:    AG # # # # # foo
      generated:               a03 AG # # # # # # # foo
                                     ^^^^^^^^^^^^^^^
                                     count these

  delta = (distance after) - (distance before)
      delta > 0  -> agent moved AWAY
      delta < 0  -> agent moved TOWARD
      delta = 0  -> agent HELD

  'foo' was never grounded. It appeared in exactly one relation: foo = STIM_A.
  If delta is reliably positive for foo, the away-function TRANSFERRED through
  the relation -- the model is doing something it was never directly taught.

CONDITIONS (all measured in the same run, same model, same prompts)
  STIM_A  grounded away stimulus     -> positive control, MUST be positive
  STIM_T  grounded toward stimulus   -> positive control, MUST be negative
  foo     hop 1 (foo = STIM_A)       -> the transfer test
  s1      hop 2 (s1 = foo)           -> decay with relational distance
  s2      hop 3 (s2 = s1)            -> decay with relational distance
  opp     opposition (opp != foo)    -> MUST INVERT to negative
  ctrl    unlinked control           -> MUST stay flat (~0)

WHY TEMPERATURE > 0
  At temperature 0 the model always emits its single most likely token and you
  measure a point estimate with no variance -- which looks suspiciously clean
  and tells you nothing about the strength of the tendency. Sampling gives a
  DISTRIBUTION of movements, so the effect has an error bar.

READING THE RESULT
  The controls carry the argument, not foo. If STIM_A is flat, training failed
  and nothing else means anything. If ctrl moves, the probe is leaking. If opp
  does NOT invert, you have association, not a relation -- which is precisely
  the alternative explanation this design exists to rule out.
"""

import torch
from tokenizer import Tokenizer, load_corpus_file
from model import GPT

MARK, AG = "#", "AG"   # MARK kept only for backward compatibility
CKPT = "model.pt"

# ---------------------------------------------------------------- settings
CONDITIONS   = ["STIM_A", "STIM_T", "STIM_N", "foo", "s1", "s2", "opp", "ctrl",
                "h00","h01","h02","h03","h04",     # held-out  = STIM_A -> AWAY
                "k00","k01","k02","k03","k04"]     # held-out != STIM_A -> TOWARD
# Probe INSIDE the range the corpus actually trained on (START_RANGE = 9..15).
# v1 probed 4..8, which after the corpus fix sits outside the trained range --
# the model would have been answering about positions it had never seen.
START_DISTS  = [9, 11, 13, 15]
REPS         = 50                # samples per (condition, start) cell
TEMPERATURE  = 0.8
# Must be long enough for the WHOLE continuation: action + AG + up to MAX_D
# marks + stimulus + one more token to confirm the mark-run ended.
# If this is too small, the samples that get discarded are the ones that moved
# FURTHEST -- selective attrition biased against the very effect being measured.
MAX_NEW      = 4                # tokens to generate before giving up
SEED         = 0


def load_model():
    ck = torch.load(CKPT)
    model = GPT(vocab_size=ck["vocab_size"], block_size=ck["block_size"],
                n_embed=ck["n_embed"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"loaded {CKPT}  (step {ck.get('step','?')}, "
          f"val loss {ck.get('val_loss', float('nan')):.4f})")
    return model, ck


@torch.no_grad()
def generate(model, ids, max_new, temperature, block_size):
    """Sample `max_new` tokens, one at a time, feeding each back in."""
    for _ in range(max_new):
        # never feed more than block_size positions (the position table's limit)
        ctx = ids[:, -block_size:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / temperature    # last position only
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def parse_next_distance(tokens):
    """Given the GENERATED tokens, find the next agent snapshot and count its
    marks. Returns None if the model never produced a well-formed snapshot."""
    try:
        k = tokens.index(AG)
    except ValueError:
        return None                      # no agent snapshot -> unusable sample
    j = k + 1
    if j >= len(tokens):
        return None
    t = tokens[j]
    if not (t.startswith("d") and t[1:].isdigit()):
        return None                      # malformed: no distance token after AG
    return int(t[1:])


def measure(model, tok, block_size, sym, d, reps, temperature):
    """Return a list of deltas for one (symbol, start-distance) cell."""
    prompt = [AG, f"d{d:02d}", sym]      # single-token distance encoding
    base = torch.tensor([tok.encode(prompt)])
    deltas, invalid = [], 0
    for _ in range(reps):
        out = generate(model, base.clone(), MAX_NEW, temperature, block_size)
        gen = tok.decode(out[0].tolist()[len(prompt):])
        nd = parse_next_distance(gen)
        if nd is None:
            invalid += 1
        else:
            deltas.append(nd - d)
    return deltas, invalid


def summarise(deltas):
    if not deltas:
        return float("nan"), float("nan"), 0
    t = torch.tensor(deltas, dtype=torch.float)
    sd = t.std().item() if len(deltas) > 1 else 0.0
    return t.mean().item(), sd, len(deltas)


def main():
    torch.manual_seed(SEED)
    corpus = load_corpus_file("corpus.txt")
    tok = Tokenizer(corpus)
    model, ck = load_model()
    block_size = ck["block_size"]

    print(f"\ntemperature {TEMPERATURE} | {REPS} reps x {len(START_DISTS)} "
          f"start distances = {REPS*len(START_DISTS)} samples per condition")
    print("\ndelta > 0 = moved AWAY   delta < 0 = moved TOWARD   0 = held\n")

    print(f"{'condition':>10s}  {'role':>22s}  {'mean':>7s}  {'sd':>6s}  "
          f"{'n':>5s}  {'bad':>4s}")
    print("-" * 66)

    roles = {
        "STIM_A": "grounded away (ctrl+)",
        "STIM_T": "grounded toward (ctrl+)",
        "STIM_N": "grounded hold (ctrl+)",
        "foo":    "hop 1 -- TRANSFER",
        "s1":     "hop 2",
        "s2":     "hop 3",
        "opp":    "opposition -- INVERT?",
        "ctrl":   "unlinked (baseline)",
    }
    for h in ["h00","h01","h02","h03","h04"]: roles[h] = "held-out = -> away"
    for k in ["k00","k01","k02","k03","k04"]: roles[k] = "held-out != -> toward"

    results = {}
    for sym in CONDITIONS:
        all_d, all_bad = [], 0
        for d in START_DISTS:
            ds, bad = measure(model, tok, block_size, sym, d, REPS, TEMPERATURE)
            all_d += ds
            all_bad += bad
        m, sd, n = summarise(all_d)
        results[sym] = (m, sd, n)
        print(f"{sym:>10s}  {roles[sym]:>22s}  {m:7.2f}  {sd:6.2f}  "
              f"{n:5d}  {all_bad:4d}")

    # ---- the three checks that decide whether this is a result ----------
    print("\n--- checks ---")
    a, t_, c = results["STIM_A"][0], results["STIM_T"][0], results["ctrl"][0]
    print(f"anchors behave?    STIM_A {a:+.2f} (want > 0), "
          f"STIM_T {t_:+.2f} (want < 0)  -> "
          f"{'PASS' if a > 0 and t_ < 0 else 'FAIL'}")
    print(f"control flat?      ctrl {c:+.2f} (want ~0)  -> "
          f"{'PASS' if abs(c) < 0.5 else 'FAIL'}")
    f, o = results["foo"][0], results["opp"][0]
    print(f"transfer?          foo {f:+.2f} (want > 0)  -> "
          f"{'PASS' if f > 0 else 'FAIL'}")
    print(f"opposition inverts? opp {o:+.2f} (want < 0)  -> "
          f"{'PASS' if o < 0 else 'FAIL'}")
    print(f"\nrelational distance: foo {results['foo'][0]:+.2f}  "
          f"s1 {results['s1'][0]:+.2f}  s2 {results['s2'][0]:+.2f}")


if __name__ == "__main__":
    main()
