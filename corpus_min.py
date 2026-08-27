"""
MINIMAL COMPLETE CORPUS  (first rung of the RFT-LLM experiment)

One agent, one stimulus per direction, one relational chain, controls.
Everything is agent-based BEHAVIOR (not coordinate placement):
    agent moves AWAY from a stimulus / TOWARD a stimulus / HOLDS (neutral).

Design safeguards baked in:
  - DEGREES OF FREEDOM: movement magnitude is VARIED (1..3), start is VARIED,
    length is VARIED  -> defeats rigid-formula memorization, gives realistic variance.
  - CLEAN ESSENTIAL LOGIC: away ALWAYS increases the gap, toward ALWAYS decreases,
    '=' ALWAYS same-behaviour, '!=' ALWAYS opposite.
  - ISOLATION: chain / test symbols appear in EXACTLY ONE relation, never grounded.
  - CONTROLS: an unrelated novel symbol (never linked to anything).
  - ONE corpus, trained together (weight integration); each unit a separate
    example (sequence isolation).

--------------------------------------------------------------------------
FOUR FIXES vs v1 (all found by the controls -- v1 FAILED the anchor check)
--------------------------------------------------------------------------
 (1) START DISTANCE WAS CONFOUNDED WITH DIRECTION.
     v1 drew away-starts from 0..6 and toward-starts from 6..12, so the
     starting gap PREDICTED the direction and the model took the shortcut
     "small gap -> grow, large gap -> shrink". It never had to read the
     stimulus at all. Proof it was a shortcut: the UNLINKED control symbol
     also moved (+2.5 at d=0, -2.0 at d=9) -- a symbol with no function
     cannot produce direction, so the direction came from the position.
     FIX: every direction now draws its start from the SAME range.

 (2) STIM_A WAS DOING DOUBLE DUTY.
     v1's make_neutral() used STIM_A as the hold-stimulus, so STIM_A meant
     "away" in 250 units and "hold" in 250 others -- its function diluted by
     half. FIX: neutral gets its own stimulus, STIM_N.

 (3) SLOT MISMATCH between what '=' was taught on and what it was tested on.
     v1's relation pairs put the throwaway symbols in the ACTION slot:
         AG # # # # # STIM_A r075 AG ...     <- r075 is an ACTION
     but the test statement equates foo with a STIMULUS:
         foo = STIM_A
     So the operation was learned in one syntactic position and probed in
     another. FIX: relation symbols now occupy the STIMULUS slot, exactly
     where STIM_A sits in a grounded unit.

 (4) THE CLAMPS WERE BINDING.
     min(MAX_D,...) / max(MIN_D,...) fired often enough to add
     regression-to-the-mean on top of everything else. FIX: the range is now
     sized so the maximum possible excursion cannot reach either bound.

Vocabulary roles:
  STIM_A  = the AWAY stimulus  (agent flees it)     -> the anchor for the chain
  STIM_T  = the TOWARD stimulus (agent approaches it)
  STIM_N  = the NEUTRAL stimulus (agent holds)      -> keeps STIM_A pure
  a00..   = away actions      t00.. = toward actions   n00.. = neutral actions
  #       = one unit of agent<->stimulus distance
  =, !=   = relations
  chain:  foo = STIM_A ; s1 = foo ; s2 = s1          (away chain, hops 1..3)
          opp != foo                                  (opposition -> toward)
  ctrl    = unrelated control (linked to nothing)
"""

import random
from collections import Counter

# ---------------------------------------------------------------- settings
MARK = "#"
SEP  = "|"
AG   = "AG"
EQ, NEQ = "=", "!="

# Range sized so the clamps can NEVER fire:
#   max excursion = STEPS_RANGE[1] * STEP_RANGE[1] = 3 * 3 = 9
#   lowest  start  9 - 9 =  0 = MIN_D   (floor never crossed)
#   highest start 15 + 9 = 24 = MAX_D   (ceiling never crossed)
MIN_D, MAX_D = 0, 24
START_RANGE  = (9, 15)       # <-- SAME for away / toward / hold   (fix 1)
STEP_RANGE   = (1, 3)        # degrees of freedom: varied movement magnitude
STEPS_RANGE  = (2, 3)        # varied sequence length

# stimuli (the things the agent moves relative to)
STIM_A = "STIM_A"            # away stimulus (and the chain anchor)
STIM_T = "STIM_T"            # toward stimulus
STIM_N = "STIM_N"            # neutral stimulus                    (fix 2)

# action repertoires
AWAY_ACTIONS    = [f"a{i:02d}" for i in range(6)]
TOWARD_ACTIONS  = [f"t{i:02d}" for i in range(6)]
NEUTRAL_ACTIONS = [f"n{i:02d}" for i in range(6)]

# throwaway pool for teaching the = / != operations
REL_POOL = [f"r{i:03d}" for i in range(120)]

# HELD-OUT PROBES (the manipulation check).
# These appear ONLY in a statement relating them to a grounded anchor -- never
# demonstrated, never grounded. Structurally identical to 'foo', but there are
# ten of them with KNOWN predicted directions, so the operation can be tested
# with statistical power without touching the real test symbols.
HELD_EQ  = [f"h{i:02d}" for i in range(5)]    # h.. = STIM_A   -> predict AWAY
HELD_NEQ = [f"k{i:02d}" for i in range(5)]    # k.. != STIM_A  -> predict TOWARD


def marks(d):
    """DISTANCE AS A SINGLE TOKEN.

    v2 wrote distance in UNARY -- twelve '#' for a distance of twelve. That made
    '#' 75.5% of the corpus, and since cross-entropy averages over every token
    position equally, the loss became almost entirely a measure of how well the
    model counts marks. '=' and '!=' were 1.59% of tokens, so ignoring them cost
    the optimiser essentially nothing -- and it did ignore them (the operator
    failed even in-distribution: '=' and '!=' produced the same behaviour).

    One token per distance puts the relation operators back where they can
    actually influence the loss, and shrinks block_size by ~4x as a bonus."""
    return [f"d{d:02d}"]

def _actions_for(direction):
    return {"away": AWAY_ACTIONS, "toward": TOWARD_ACTIONS,
            "hold": NEUTRAL_ACTIONS}[direction]

def _apply(direction, d, step):
    """Move the gap. The clamps are guards only -- the ranges above guarantee
    they never actually fire (verify() asserts this)."""
    if direction == "away":   return min(MAX_D, d + step)
    if direction == "toward": return max(MIN_D, d - step)
    return d

def _opp(direction):
    return {"away": "toward", "toward": "away"}[direction]


# ---------------------------------------------------------------- PART 1: grounded behaviour
def _snapshot_pair(stimulus, direction, d):
    """One agent snapshot -> action -> next agent snapshot, relative to
    `stimulus`. Returns (tokens, new_distance).

    THE CANONICAL SLOT ORDER, used identically everywhere in the corpus:
        AG <d marks> STIM <action> AG <nd marks> STIM
                     ^stimulus     ^action
    """
    act = random.choice(_actions_for(direction))
    step = random.randint(*STEP_RANGE)
    nd = _apply(direction, d, step)
    toks = [AG] + marks(d) + [stimulus, act, AG] + marks(nd) + [stimulus]
    return toks, nd

def _behavior(stimulus, direction):
    """Agent moves relative to `stimulus`. direction: 'away'/'toward'/'hold'.
    Magnitude / start / length all VARIED, but the START is drawn from the same
    range whatever the direction, so it leaks nothing about which way to move."""
    n = random.randint(*STEPS_RANGE)
    d = random.randint(*START_RANGE)          # <-- fix 1: direction-independent
    toks = []
    for _ in range(n):
        part, d = _snapshot_pair(stimulus, direction, d)
        toks += part
    toks += [SEP]
    return toks

def make_away():    return _behavior(STIM_A, "away")
def make_toward():  return _behavior(STIM_T, "toward")
def make_neutral(): return _behavior(STIM_N, "hold")     # <-- fix 2: own stimulus


# ---------------------------------------------------------------- PART 2: relation-operation training
def make_relation_pair(relation, direction):
    """A rel B, then show A behaving (direction) and B behaving (same if '=',
    opposite if '!='). The throwaway symbols A and B sit in the STIMULUS slot,
    exactly where STIM_A sits in a grounded unit, so the operation is learned in
    the same syntactic position the test statements will use (fix 3)."""
    A, B = random.sample(REL_POOL, 2)

    dA = random.randint(*START_RANGE)
    partA, _ = _snapshot_pair(A, direction, dA)

    dirB = direction if relation == EQ else _opp(direction)
    dB = random.randint(*START_RANGE)
    partB, _ = _snapshot_pair(B, dirB, dB)

    return [A, relation, B, SEP] + partA + [SEP] + partB + [SEP]


# ------------------------------------------- PART 2b: ANCHORED derivation
# WHY THIS EXISTS (the gap the first measurement exposed):
#   make_relation_pair() puts BOTH premises in the same sequence -- the
#   statement AND a demonstration of A. So the model learns
#       "statement + a demonstration of A  ->  predict B"
#   which attention can read straight off the context.
#   But the TEST gives a statement and NO demonstration:
#       foo = STIM_A |
#   STIM_A's function was established in OTHER units, so deriving foo requires
#   integrating ACROSS units, through the weights. The model was never trained
#   on that operation -- which is why foo sat at chance while the anchors were
#   pristine.
#
#   These units train exactly that: A related to a GROUNDED anchor which is NOT
#   demonstrated here, then A behaves. Multiple-exemplar training for the
#   cross-unit derivation, on throwaway symbols. The real test symbols stay
#   untouched and ungrounded.

ANCHOR_DIR = {STIM_A: "away", STIM_T: "toward"}

def make_anchored_pair(relation, anchor):
    """  A rel ANCHOR | <A behaves>  -- the anchor never behaves in this unit."""
    A = random.choice(REL_POOL)
    base = ANCHOR_DIR[anchor]
    dirA = base if relation == EQ else _opp(base)
    d = random.randint(*START_RANGE)
    part, _ = _snapshot_pair(A, dirA, d)
    return [A, relation, anchor, SEP] + part + [SEP]

# ---------------------------------------------------------------- PART 3: test statements (isolated)
def make_test_units():
    """One away-chain from STIM_A, plus an opposition branch, plus an unrelated
    control. Each test symbol appears in EXACTLY ONE relation. Consequences
    WITHHELD -- the statement is given, the behaviour never is.
    'ctrl' appears ALONE (never in a relation, never grounded) -> pure baseline:
    the model gets an embedding for it but learns no function for it."""
    units, info = [], []
    units.append(["foo", EQ,  STIM_A, SEP]); info.append(("foo", 1, "away"))
    units.append(["s1",  EQ,  "foo",  SEP]); info.append(("s1",  2, "away"))
    units.append(["s2",  EQ,  "s1",   SEP]); info.append(("s2",  3, "away"))
    units.append(["opp", NEQ, "foo",  SEP]); info.append(("opp", None, "toward"))
    units.append(["ctrl", SEP]);             info.append(("ctrl", None, "none"))
    # held-out probes: statement only, never demonstrated (manipulation check)
    for h in HELD_EQ:
        units.append([h, EQ, STIM_A, SEP]);  info.append((h, 1, "away"))
    for k in HELD_NEQ:
        units.append([k, NEQ, STIM_A, SEP]); info.append((k, 1, "toward"))
    return units, info


# ---------------------------------------------------------------- assemble
def build_corpus(n_away=200, n_toward=200, n_neutral=200,
                 n_relations=1600, n_anchored=1600, repeat_tests=40, seed=0):
    random.seed(seed)
    corpus = []
    for _ in range(n_away):    corpus.append(make_away())
    for _ in range(n_toward):  corpus.append(make_toward())
    for _ in range(n_neutral): corpus.append(make_neutral())

    per = n_relations // 4
    for _ in range(per): corpus.append(make_relation_pair(EQ,  "away"))
    for _ in range(per): corpus.append(make_relation_pair(EQ,  "toward"))
    for _ in range(per): corpus.append(make_relation_pair(NEQ, "away"))
    for _ in range(per): corpus.append(make_relation_pair(NEQ, "toward"))

    per_a = n_anchored // 4
    for _ in range(per_a): corpus.append(make_anchored_pair(EQ,  STIM_A))
    for _ in range(per_a): corpus.append(make_anchored_pair(EQ,  STIM_T))
    for _ in range(per_a): corpus.append(make_anchored_pair(NEQ, STIM_A))
    for _ in range(per_a): corpus.append(make_anchored_pair(NEQ, STIM_T))

    test_units, info = make_test_units()
    for _ in range(repeat_tests):
        for u in test_units:
            corpus.append(list(u))

    random.shuffle(corpus)
    return corpus, info


# ---------------------------------------------------------------- verification
def _distances(unit):
    """The sequence of agent distances (marks immediately after each AG)."""
    dists, i = [], 0
    while i < len(unit):
        if unit[i] == AG and i + 1 < len(unit) and unit[i+1].startswith("d") \
           and unit[i+1][1:].isdigit():
            dists.append(int(unit[i+1][1:])); i += 2
        else:
            i += 1
    return dists

def _transitions(unit):
    """The REAL distance changes in a unit, as (before, after) pairs.

    CAREFUL -- this is where v1's verify() was wrong. Each snapshot pair emits
    TWO agent positions (d then nd), and the NEXT pair starts again at nd:

        distances: [13, 14, 14, 17]
                         ^^^^^^ the same moment, written twice

    zip(d, d[1:]) would count that 14->14 as a 'hold' that never happened and
    report ~37% phantom neutrals. Every pair contributes exactly two AGs, so
    the real transitions are the even entries against the odd ones."""
    d = _distances(unit)
    return list(zip(d[0::2], d[1::2]))

def _stimulus_of(unit):
    """The token in the STIMULUS slot: first token after the first mark-run."""
    if AG not in unit:
        return None
    i = unit.index(AG) + 1
    if i < len(unit) and unit[i].startswith("d") and unit[i][1:].isdigit():
        i += 1
    return unit[i] if i < len(unit) else None

def verify(corpus, info):
    print(f"total units: {len(corpus)}")

    U = D = S = 0
    for u in corpus:
        for a, b in _transitions(u):
            if b > a: U += 1
            elif b < a: D += 1
            else: S += 1
    print(f"direction balance -> away(up):{U}  toward(down):{D}  neutral(same):{S}")

    # ---- FIX-1 CHECK: does the STARTING distance predict the direction? ----
    # Only the FIRST transition of each unit is checked. Later transitions
    # inevitably drift (an agent that has already fled twice IS further out),
    # but that drift is a CONSEQUENCE of behaving, not a clue planted in the
    # corpus -- and the measurement probe presents one snapshot and reads the
    # FIRST move, so the first transition is the one that must be clean.
    first, later = {"up": [], "down": [], "same": []}, {"up": [], "down": [], "same": []}
    for u in corpus:
        for n, (a, b) in enumerate(_transitions(u)):
            key = "up" if b > a else ("down" if b < a else "same")
            (first if n == 0 else later)[key].append(a)
    print("\n[fix 1] start-distance by direction, FIRST transition "
          "(MUST be ~equal, else confounded):")
    for k in ("up", "down", "same"):
        v = first[k]
        if not v: continue
        print(f"  {k:5s}: mean start {sum(v)/len(v):5.2f}   "
              f"min {min(v):2d}  max {max(v):2d}  n={len(v)}")
    means = [sum(first[k])/len(first[k]) for k in first if first[k]]
    spread = max(means) - min(means)
    print(f"  spread between means: {spread:.2f}   "
          f"-> {'PASS' if spread < 0.5 else 'FAIL'} (want < 0.5)")
    lm = [sum(later[k])/len(later[k]) for k in later if later[k]]
    if lm:
        print(f"  (later transitions spread {max(lm)-min(lm):.2f} -- expected "
              f"drift, not a leak)")

    # ---- FIX-2 CHECK: is each stimulus tied to exactly one function? -------
    stim_dir = {}
    for u in corpus:
        s_ = _stimulus_of(u)
        if s_ not in (STIM_A, STIM_T, STIM_N):
            continue
        for a, b in _transitions(u):
            key = "away" if b > a else ("toward" if b < a else "hold")
            stim_dir.setdefault(s_, Counter())[key] += 1
    print("\n[fix 2] stimulus -> function purity (each should be 100% one way):")
    pure = True
    for s, c in sorted(stim_dir.items()):
        tot = sum(c.values())
        top = c.most_common(1)[0][1] / tot
        pure &= (top > 0.99)
        parts = "  ".join(f"{k} {v/tot:5.1%}" for k, v in c.most_common())
        print(f"  {s:8s}: {parts}   (n={tot})")
    print(f"  -> {'PASS' if pure else 'FAIL'}")

    # ---- FIX-3 CHECK: relation symbols in the STIMULUS slot? --------------
    rel_units = [u for u in corpus if len(u) > 3 and u[1] in (EQ, NEQ)
                 and u[0].startswith("r")]
    ok = sum(1 for u in rel_units if _stimulus_of(u) == u[0])
    print(f"\n[fix 3] relation symbol occupies the STIMULUS slot: "
          f"{ok}/{len(rel_units)}  -> {'PASS' if ok == len(rel_units) else 'FAIL'}")

    # ---- FIX-4 CHECK: did any clamp fire? --------------------------------
    hit = sum(1 for u in corpus for d in _distances(u) if d in (MIN_D, MAX_D))
    print(f"\n[fix 4] clamp hits (distance sitting exactly on a bound): {hit}  "
          f"-> {'PASS' if hit == 0 else 'FAIL'} (want 0)")

    # ---- isolation --------------------------------------------------------
    flat = Counter()
    for u in corpus: flat.update(u)
    print("\ntest-symbol appearances:")
    for sym in ["foo", "s1", "s2", "opp", "ctrl"]:
        print(f"  {sym:5s}: {flat.get(sym,0)}")

    print("\nheld-out probe symbols (statement only, NEVER demonstrated):")
    hcount = {h: flat.get(h, 0) for h in HELD_EQ + HELD_NEQ}
    print(f"  {hcount}")
    h_leak = any(AG in u and (set(HELD_EQ + HELD_NEQ) & set(u)) for u in corpus)
    print(f"  held-out symbols demonstrated anywhere? {h_leak}  "
          f"-> {'PASS' if not h_leak else 'FAIL'}")

    n_anch = sum(1 for u in corpus if len(u) > 3 and u[1] in (EQ, NEQ)
                 and u[2] in (STIM_A, STIM_T) and AG in u)
    print(f"\nanchored-derivation units (statement + behaviour, anchor NOT shown): {n_anch}")

    test_syms = {"foo", "s1", "s2", "opp", "ctrl"}
    leak = any(AG in u and (test_syms & set(u)) for u in corpus)
    print(f"\ntest symbols leaked into grounded behaviour units? {leak}  "
          f"-> {'PASS' if not leak else 'FAIL'}")

    print("\nkey grounded symbols:")
    for sym in [STIM_A, STIM_T, STIM_N, "a00", "t00", "n00"]:
        print(f"  {sym:8s}: {flat.get(sym,0)}")

    longest = max(len(u) for u in corpus)
    print(f"\nlongest unit: {longest} tokens  -> block_size will be {longest-1}")


# ---------------------------------------------------------------- save to file
def save_corpus(corpus, path="corpus.txt"):
    """Write the corpus to a text file: ONE UNIT PER LINE, tokens space-separated.
    This makes the material PERSIST on disk so training can read it later.
    (Without this the corpus lives only in memory and vanishes on exit.)"""
    with open(path, "w") as f:
        for unit in corpus:
            f.write(" ".join(unit) + "\n")
    print(f"saved {len(corpus)} units to {path}")


if __name__ == "__main__":
    print("=== sample away behaviour ===")
    random.seed(3); print(" ".join(make_away()))
    print("\n=== sample toward behaviour ===")
    print(" ".join(make_toward()))
    print("\n=== sample neutral behaviour ===")
    print(" ".join(make_neutral()))
    print("\n=== sample relation (=, away) ===")
    print(" ".join(make_relation_pair(EQ, "away")))
    print("\n=== sample relation (!=, toward) ===")
    print(" ".join(make_relation_pair(NEQ, "toward")))
    print("\n=== test units ===")
    tu, info = make_test_units()
    for u in tu: print(" ".join(u))

    print("\n=== VERIFY FULL CORPUS ===")
    corpus, info = build_corpus()
    verify(corpus, info)

    print("\n=== SAVE ===")
    save_corpus(corpus, "corpus.txt")
