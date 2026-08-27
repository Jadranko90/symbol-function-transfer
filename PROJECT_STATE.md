# Project State — working notes

Detailed record behind the README: what was decided and why, what is known to be
wrong, and what to do next. Written so the project can be picked up cold.

# STUDY 2 — the positive result

## Design
Two response domains. A symbol is trained in one and tested in the other.

| context | form | who appears |
|---|---|---|
| ANNOUNCEMENT | `foo ! ! ! !` | everyone |
| MOVEMENT | `AG ## p058 a02 AG ##### p058` | bridge symbols + `STIM_A` only |

- **Bridge** — 102 symbols, each with an intensity 1–6, appearing in **both**
  contexts (announces N, flees N). This teaches that announcement-count and
  movement-magnitude are the same dimension. Without it there is no link between
  the domains and the test is unanswerable.
- **Anchor** — `STIM_A` announces 2, appears in both.
- **Test symbols** — `foo` (4), `bar` (6), `q1..q6` (1..6): **announcement only,
  never once in a movement unit.**
- **Baseline** — `ctrl`: bare, no announcements, no movement, no information.
- **No zeros anywhere.** No neutral stimulus, no hold behaviour, every movement
  ≥ 1. "Stay put" is not a trained response, so the agent always moves. Achieved
  by omission — an explicit "novel → move randomly" rule would be a COMPETING
  contingency and could mask the transfer.

## Result (6 independent seeds — fresh corpus, init and training each)

> **A symbol trained only to emit announcements acquires a graded movement
> function it was never taught, in a response domain it never appeared in, with
> no supporting context in the prompt.**

| test | mean ± sd | t | p | d |
|---|---|---|---|---|
| **slope (announced → moved)** | **0.61 ± 0.14** | 10.55 | **.0001** | 4.31 |
| **r (announced vs moved)** | **0.81 ± 0.07** | 28.77 | **<.0001** | 11.75 |
| foo − ctrl | 1.66 ± 2.04 | 1.99 | .0516 | 0.81 |
| bar − ctrl | 2.36 ± 2.29 | 2.52 | .0265 | 1.03 |
| bar − foo (ordering) | 0.71 ± 1.23 | 1.41 | .1087 | 0.58 |

Dose-response, **monotonic across all six levels**, all held out of movement:

| announced | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| moved | 1.98 | 3.43 | 3.65 | 3.92 | 5.07 | 5.18 |

Gate passed: `STIM_A` (trained, announces 2) moved **2.00 ± 0.00**.
Raw means: `STIM_A` 2.00, `foo` 4.29 (announced 4), `bar` 5.00 (announced 6),
`ctrl` 2.64.

**Replicated on a second machine** with a different PyTorch build: slope 0.59 →
0.61, r 0.84 → 0.81, `foo` 4.01 → 4.29. Note the first single-seed run gave
slope 0.31 and r 0.68 with a non-monotonic curve — replication changed the
numbers materially, it did not merely confirm them.

## Statistics — the unit of analysis
**Seed is the unit.** Each seed is an independent replication (fresh corpus,
fresh init, fresh training) and contributes ONE slope, ONE r, ONE contrast.
One-sample t-tests across seeds, n = 6, one-tailed.

Do NOT pool the ~600 generations per seed and regress. Generations from the same
symbol on the same model are many samples of ONE thing, not many independent
observations; that would inflate n and produce a meaningless p-value.

Two reporting notes: r values are averaged directly (Fisher z changes the mean
by <.01 in this range), and `d` on the slope and r rows is arithmetically correct
but implausible-looking — quote `d` only for the contrasts, where it explains why
a large effect missed significance.

## Known issues
- **`ctrl` contrast is BIMODAL across seeds**: +3.35, −0.46, +3.93, −0.62, +3.10,
  +0.64. Three seeds show a large effect, three show none. **This reproduced on a
  second machine with the same seed parity** — so it is systematic and findable,
  not noise. Currently unexplained. Most likely cause: `ctrl` appears in only 60
  trivial units, so its embedding barely moves from initialisation and lands
  arbitrarily in the intensity direction (SD 1.57, against `STIM_A`'s 0.00).
- **Compression at both ends.** Announced 1 moved 1.98, announced 6 moved 5.18;
  everything pulls toward ~3.5, the mean of the trained movement distribution.
  Slope 0.61 not 1.0. Middle of the curve has the highest variance (sd 1.15,
  1.22 at announced 3–4 against 0.35, 0.52 at 5–6).
- Report slope and r as primary; `ctrl` contrasts as secondary with the
  instability disclosed.

---

# STUDY 1 — the null (completed, 5 seeds)

Related a novel symbol to a grounded one with an operator: `foo = STIM_A`.

> **Derived relational responding occurs within the context window. It leaves no
> durable trace in the symbol's representation.**

With the statement in the prompt: `foo = STIM_A` → **+1.96**, `foo != STIM_A` →
**−1.32** (inverts), `foo = STIM_T` → **−2.00** (tracks which anchor). Held-out
probes `h..` +1.74, `k..` −1.69. Replicated across 5 seeds. Both alternative
explanations excluded — opposition inverts (rules out association), anchor
identity matters (rules out generic priming).

With the statement removed: `foo` −0.31, `ctrl` −0.13, `h..` +0.08, `k..` +0.28.
Nothing.

**Why the null is structural, not empirical.** The loss on `foo = STIM_A |` asks
only about the next token. No term in that objective references movement, so the
gradient with respect to "how far the agent moves near foo" is exactly **zero**.
No amount of exposure can write a behavioural function into a symbol the
objective never asks to behave.

**Corollary:** any procedure that writes a movement function into `foo` requires
training on `foo` moving — which IS grounding. So durable transformation and
derivation-from-a-bare-statement are mutually exclusive by construction. This
killed a proposed "self-generated consolidation" study before it was built.

**Caveat on the null.** Study 1's corpus HAD a hold option (`STIM_N`), so a bare
symbol could default to holding. The flat result is therefore consistent with
both "no function transferred" and "the model defaulted to hold." Untested.
Deleting `STIM_N` and re-probing would settle it — and could refute the
zero-gradient explanation above, since `foo` is trained to predict `STIM_A` and
may sit near it in embedding space as a side effect.

---

# BUGS FOUND (every one caught by a control, none by reading code)

1. **Start-distance confound** — away-starts 0..6, toward-starts 6..12, so
   position predicted direction. Giveaway: the *unlinked control* moved
   (+2.5 at d=0, −2.0 at d=9).
2. **STIM_A double duty** — neutral units used `STIM_A`, diluting its function 50%.
3. **Slot mismatch** — relations taught on symbols in the ACTION slot, tested on
   a STIMULUS.
4. **Binding clamps** — min/max fired often, adding regression to the mean.
5. **verify() parser bug** — snapshot boundaries counted as phantom "hold"
   transitions (~40% fiction in the direction balance).
6. **Selective attrition** — `MAX_NEW` truncation discarded the LARGEST movements.
7. **THE BIG ONE — unary encoding drowned the loss.** `#` was 75.5% of Study 1's
   corpus; `=`/`!=` were 1.59%. The loss was almost entirely "can you count."
   The model looked excellent (val 0.258, anchors perfect, 100% legal
   generations) while ignoring the operators entirely — operator separation
   −0.38. Single-token distances took it to **+7.07**.

   **In a controlled corpus, token frequency IS loss weighting. Check the token
   histogram before trusting any val loss.**

   Study 2 uses unary `#` (50.6%) deliberately — there the marks ARE the
   dependent variable, so frequency is signal, not noise.

---

# NEXT STEPS

Ordered by value. None are required — the result stands as published.

1. **Fix the `ctrl` baseline.** Highest value, smallest change. Replace the empty
   symbol with one that announces a fixed mid intensity but never moves. Anchored
   baseline instead of one floating on initialisation noise; both contrasts would
   likely go significant without touching the design.
2. **Chase the even/odd seed split.** Reproduces across machines, so it is real
   and findable. Probably an hour. "Found an anomaly, tracked it down" is a
   better story than "unexplained."
3. **Widen intensity to 1–10.** Separates regression-to-the-mean from a genuine
   ceiling: under regression the curve stretches, under a ceiling it flattens.
   **Caveat:** a wider training range moves the mean everything regresses toward,
   so compression may spread proportionally rather than shrink.
4. **Held-out interpolation** — train on 1–4 and 8–10, leave 5–7 unseen. If the
   model places unseen intermediate values correctly, that is much stronger
   evidence of a continuous dimension than denser sampling of trained values.
   Cheaper and more informative than (3).
5. **Batch the measurement.** ~600 sequential generations per seed is most of the
   wall-clock time. Batched, 20 seeds would run faster than 6 do now — which
   fixes the underpowered contrasts directly.
6. **Add a direction dimension.** Everything currently flees, so only magnitude
   was tested. A reviewer can ask whether `foo` acquired *aversiveness* or just
   *bigness*. Some symbols announcing with a different marker and the agent
   approaching would settle it.
7. **Ordering test between distant symbols** — announce 1 vs 10 rather than 4 vs
   6. The current `bar − foo` null is a design choice about where the test was
   placed, not a finding.
8. **Embedding-direction check** (~25 lines). Regress bridge embeddings against
   intensity; see whether a single direction predicts it, and whether `foo`,
   `bar` and `ctrl` sit where their announcements say. Would connect the
   behavioural result to the linear-representation literature and probably
   explain `ctrl` at the same time.
9. **Mutual entailment** — `foo` announces more than `STIM_A`; does the reverse
   relation hold? Classic RFT criterion, never tested.

---

# POSITIONING (decided)

- **Not a new phenomenon; a new preparation.** The mechanism is probably the
  linear-representation phenomenon behind `king − man + woman ≈ queen`. What is
  new is that it is measured *behaviourally*, on a corpus where every exposure is
  auditable, in a transformer.
- **"Movement" is a description of the encoding, not a claim about
  representation.** What is measured is a graded response class under symbolic
  control.

---

# ENVIRONMENT

CPU only, no GPU. `pip install torch scipy`.
`corpus2.py` → `train.py` (~60s) → `measure2.py`, or `seeds2.py` for the full
result (~15 min). Study 1 and Study 2 both write `corpus.txt` — delete it between
studies so a stale corpus cannot be trained on silently.
