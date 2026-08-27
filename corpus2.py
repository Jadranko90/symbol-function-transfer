"""Study 2 corpus -- transformation of stimulus function across response domains.

THE QUESTION
  Does a symbol acquire a graded behavioural function in a domain it never
  appeared in, purely from its position on a dimension learned elsewhere?

TWO DOMAINS
  ANNOUNCEMENT   foo ! ! ! !                    a symbol emits a quantity
  MOVEMENT       AG ## p058 a02 AG ##### p058   an agent flees a stimulus

THE BRIDGE
  Nothing intrinsically links "announces 4" to "flees 4". That link is taught on
  OTHER symbols: 102 bridge symbols each have an intensity and appear in BOTH
  domains, announcing that many marks and fleeing that many marks. This is what
  makes announcement-count and movement-magnitude one dimension.

THE TEST
  foo (4), bar (6) and q1..q6 (1..6) appear ONLY in announcements -- never once
  in a movement unit. ctrl appears bare, with no announcements and no movement,
  as the floor. If movement magnitude tracks announced intensity for symbols that
  have never moved, the function transferred.

NO ZEROS
  There is no neutral stimulus and no hold behaviour; every movement is at least
  1. "Stay put" is not a trained response, so the agent cannot decline to reveal
  what a symbol encodes. Achieved by omission rather than by an explicit
  "move randomly when unfamiliar" rule, which would be a competing contingency
  and could mask the transfer being measured.
"""

import random
from collections import Counter

MARK = "#"
BANG = "!"
SEP = "|"
AG = "AG"

MIN_D, MAX_D = 0, 24
START_RANGE = (2, 8)
INTENSITIES = (1, 6)
ACTIONS = [f"a{i:02d}" for i in range(6)]

BRIDGE = [f"p{i:03d}" for i in range(102)]
BRIDGE_INTENSITY = {}

STIM_A, STIM_A_INT = "STIM_A", 2
FOO, FOO_INT = "foo", 4
BAR, BAR_INT = "bar", 6
PROBES = {f"q{i}": i for i in range(1, 7)}
CTRL = "ctrl"


def marks(n): return [MARK] * n
def bangs(n): return [BANG] * n


def make_announce(sym, intensity):
    return [sym] + bangs(intensity) + [SEP]


def make_move(sym, intensity):
    """Start distance and action vary; the MAGNITUDE is the symbol's intensity."""
    d = random.randint(*START_RANGE)
    nd = min(MAX_D, d + intensity)
    return [AG] + marks(d) + [sym, random.choice(ACTIONS), AG] + marks(nd) + [sym, SEP]


def build_corpus(n_bridge_announce=14, n_bridge_move=14,
                 n_anchor=400, repeat_tests=60, seed=0):
    random.seed(seed)
    corpus = []

    lo, hi = INTENSITIES
    levels = [lo + (i % (hi - lo + 1)) for i in range(len(BRIDGE))]
    random.shuffle(levels)
    BRIDGE_INTENSITY.clear()
    BRIDGE_INTENSITY.update(dict(zip(BRIDGE, levels)))

    for sym in BRIDGE:
        inten = BRIDGE_INTENSITY[sym]
        for _ in range(n_bridge_announce):
            corpus.append(make_announce(sym, inten))
        for _ in range(n_bridge_move):
            corpus.append(make_move(sym, inten))

    for _ in range(n_anchor):
        corpus.append(make_announce(STIM_A, STIM_A_INT))
        corpus.append(make_move(STIM_A, STIM_A_INT))

    for _ in range(repeat_tests):
        corpus.append(make_announce(FOO, FOO_INT))
        corpus.append(make_announce(BAR, BAR_INT))
        for q, inten in PROBES.items():
            corpus.append(make_announce(q, inten))
        corpus.append([CTRL, SEP])

    random.shuffle(corpus)
    return corpus


def _is_move(u): return AG in u


def verify(corpus):
    """Checks that must pass before any result is meaningful."""
    print(f"total units: {len(corpus)}")
    flat = Counter()
    for u in corpus:
        flat.update(u)

    tests = {FOO, BAR, CTRL} | set(PROBES)
    leaked = [s for s in tests if any(_is_move(u) and s in u for u in corpus)]
    print(f"\ntest symbols in a MOVEMENT unit? {leaked if leaked else False}  "
          f"-> {'PASS' if not leaked else 'FAIL'}")

    zeros, mags = 0, Counter()
    for u in corpus:
        if not _is_move(u):
            continue
        runs, i = [], 0
        while i < len(u):
            if u[i] == AG:
                j, n = i + 1, 0
                while j < len(u) and u[j] == MARK:
                    n += 1
                    j += 1
                runs.append(n)
                i = j
            else:
                i += 1
        if len(runs) >= 2:
            delta = runs[1] - runs[0]
            mags[delta] += 1
            if delta <= 0:
                zeros += 1
    print(f"movements with magnitude <= 0: {zeros}  -> "
          f"{'PASS' if zeros == 0 else 'FAIL'} (agent always moves)")
    print(f"movement magnitudes: {dict(sorted(mags.items()))}")

    print(f"\nbridge symbols: {len(BRIDGE)}   intensity spread: "
          f"{dict(sorted(Counter(BRIDGE_INTENSITY.values()).items()))}")

    # Token balance. An earlier version of this experiment failed because one
    # token was 75% of the corpus, so cross-entropy was almost entirely
    # measuring that token and the signal-carrying tokens had no influence on
    # the loss. Printed every build so it cannot happen unnoticed.
    tot = sum(flat.values())
    print(f"\ntoken balance:")
    for t in [MARK, BANG, AG, SEP]:
        print(f"  {t:4s} {flat[t]:7,}  {flat[t]/tot:6.1%}")
    longest = max(len(u) for u in corpus)
    print(f"\ntotal tokens: {tot:,}   longest unit: {longest} "
          f"-> block_size {longest-1}")


def save_corpus(corpus, path="corpus.txt"):
    with open(path, "w") as f:
        for u in corpus:
            f.write(" ".join(u) + "\n")
    print(f"saved {len(corpus)} units to {path}")


if __name__ == "__main__":
    random.seed(1)
    print("=== announcements ===")
    print(" ".join(make_announce(STIM_A, STIM_A_INT)), "  <- anchor, 2")
    print(" ".join(make_announce(FOO, FOO_INT)), "  <- foo, 4")
    print(" ".join(make_announce(BAR, BAR_INT)), "  <- bar, 6")
    print("\n=== movement (bridge + anchor only) ===")
    print(" ".join(make_move("p001", 2)))
    print(" ".join(make_move(STIM_A, STIM_A_INT)))

    print("\n=== VERIFY ===")
    c = build_corpus()
    verify(c)
    print("\n=== SAVE ===")
    save_corpus(c, "corpus.txt")
