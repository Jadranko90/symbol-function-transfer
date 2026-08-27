# RFT-LLM Experiment — Project State (handoff)

Paste this at the start of a new chat, and upload the .py files, to continue.

## What this is
A controlled experiment testing whether a small transformer shows **derived
relational responding** and **transformation of stimulus function** (Relational
Frame Theory / behavior analysis). The model is a behavioral subject: its entire
training corpus is controlled, then a symbol is tested for a function it was
never directly taught.

There are now **two studies**, and they make a coherent pair.

---

# STUDY 2 — THE POSITIVE RESULT (current work)

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
  never once in a movement unit**.
- **Baseline** — `ctrl`: bare, no announcements, no movement, no information.
- **No zeros anywhere.** No neutral stimulus, no hold behavior, every movement
  ≥ 1. "Stay put" is not a trained response, so the agent always moves. Achieved
  by omission — an explicit "novel → move randomly" rule would be a COMPETING
  contingency and could mask the transfer.

## Result (6 independent seeds — fresh corpus, init, and training each)

> **A symbol trained only to emit announcements acquires a graded movement
> function it was never taught, in a response domain it never appeared in, with
> no supporting context in the prompt.**

| test | mean ± sd | t | p | d |
|---|---|---|---|---|
| **slope (announced → moved)** | **0.59 ± 0.18** | 8.04 | **.0002** | 3.28 |
| **r (announced vs moved)** | **0.84 ± 0.02** | 91.9 | **<.0001** | 37.5 |
| foo − ctrl | 1.50 ± 1.75 | 2.10 | .045 | 0.86 |
| bar − ctrl | 2.40 ± 2.22 | 2.65 | .023 | 1.08 |
| bar − foo (ordering) | 0.91 ± 1.27 | 1.74 | .071 | 0.71 |

Dose-response, **monotonic across all six levels**, all held out of movement:

| announced | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| moved | 1.99 | 3.03 | 3.40 | 3.66 | 4.66 | 5.07 |

Gate passed: `STIM_A` (trained, announces 2) moved **2.00 ± 0.00**.
Raw means: `STIM_A` 2.00, `foo` 4.01 (announced 4), `bar` 4.92 (announced 6),
`ctrl` 2.52.

## Statistics — the unit of analysis
**Seed is the unit.** Each seed is an independent replication (fresh corpus,
fresh init, fresh training) and contributes ONE slope, ONE r, ONE contrast.
One-sample t-tests across seeds, n = 6, one-tailed.

**Do NOT pool the ~600 generations per seed and regress.** Generations from the
same symbol on the same model are 160 samples of ONE thing, not 160 independent
observations. Treating clustered data as independent inflates n and makes
anything look significant.

## Known issues in Study 2
- **`ctrl` contrast is BIMODAL across seeds**: +3.60, −0.01, +3.02, −0.22,
  +2.57, +0.02. Three seeds show a large effect, three show none. Not ordinary
  noise — looks like two regimes. Likely `ctrl` itself: with no information its
  behavior is arbitrary and varies wildly (sd 1.25 vs `STIM_A`'s 0.00). The
  dose-response avoids this by averaging six symbols and never referencing `ctrl`.
  **Fix to try:** replace the empty baseline with a symbol that announces a fixed
  mid intensity but never moves.
- **Compression at the top end**: `bar` announced 6, moved 4.92. Slope 0.59, not
  1.0. Could be regression to the mean of the movement distribution, or a bridge
  too coarse (17 exemplars per level). Widening the range to 1–12, or adding
  symbols per level, would separate those.
- Report slope and r as primary; the `ctrl` contrasts as secondary with the
  instability disclosed.

---

# STUDY 1 — THE NULL (completed, 5 seeds)

Related a novel symbol to a grounded one with an operator: `foo = STIM_A`.

> **Derived relational responding occurs within the context window. It leaves no
> durable trace in the symbol's representation.**

With the statement in the prompt: `foo = STIM_A` → **+1.96**, `foo != STIM_A` →
**−1.32** (inverts), `foo = STIM_T` → **−2.00** (tracks which anchor). Held-out
probes `h..` +1.74, `k..` −1.69. All replicated across 5 seeds. Both alternative
explanations excluded — opposition inverts (rules out association), anchor
identity matters (rules out generic priming).

With the statement removed: `foo` −0.31, `ctrl` −0.13, `h..` +0.08, `k..` +0.28.
Nothing. The two held-out groups should be opposite in sign and are not
distinguishable.

**Why the null is structural, not empirical.** The loss on `foo = STIM_A |` asks
only about the next token — `=` after `foo`, `STIM_A` after `=`, `|` after that.
No term in that objective references movement, so the gradient w.r.t. "how far
the agent moves near foo" is exactly **zero**. No amount of exposure can write a
behavioral function into a symbol the objective never asks to behave.

**This is why Study 2 works**: there, the test symbol IS grounded — it does real
work in the announcement domain and earns a rich embedding. The question changes
from "can an empty symbol acquire a function" (impossible) to "does a function
transfer across domains" (answerable, and the answer is yes).

**Corollary worth keeping**: any procedure that writes a movement function into
`foo` requires training on `foo` moving — which IS grounding. So durable
transformation and derivation-from-a-bare-statement are mutually exclusive *by
construction*, not as a finding.

---

# BUGS FOUND (every one caught by a control, none by reading code)

**The controls are the method.** Arguably as publishable as the results.

1. **Start-distance confound** — v1 drew away-starts 0..6, toward-starts 6..12,
   so position predicted direction. Giveaway: the *unlinked control* moved
   (+2.5 at d=0, −2.0 at d=9). A symbol with no function cannot produce direction.
2. **STIM_A double duty** — neutral units used `STIM_A`, diluting its function 50%.
3. **Slot mismatch** — relations taught on symbols in the ACTION slot, tested on
   a STIMULUS. Operation learned in one syntactic position, probed in another.
4. **Binding clamps** — min/max fired often, adding regression to the mean.
5. **verify() parser bug** — snapshot boundaries counted as phantom "hold"
   transitions (~40% fiction in the direction balance).
6. **Selective attrition** — `MAX_NEW` truncation discarded the LARGEST movements,
   biasing against the very effect being measured. If invalid-sample rates differ
   by condition, the discards are biased.
7. **THE BIG ONE — unary encoding drowned the loss.** `#` was 75.5% of Study 1's
   corpus; `=`/`!=` were 1.59%. Cross-entropy averages over token positions, so
   the loss was almost entirely "can you count marks." Ignoring the operators
   cost nothing and the model ignored them — while looking excellent (val 0.258,
   anchors perfect, 100% legal generations). Operator separation was −0.38.
   Single-token distances (`d13`) took it to **+7.07**.

   **In a controlled corpus, token frequency IS loss weighting. Check the token
   histogram before trusting any val loss.**

   Study 2 uses unary `#` (50.6%) deliberately and correctly — there the marks
   ARE the dependent variable, so frequency is signal, not noise.

---

## Files

**Shared infrastructure**
- `tokenizer.py` — Tokenizer + load_corpus_file. Unchanged all project.
- `batching.py` — each unit a separate padded row (isolation); targets pad −100.
- `attention.py` — `Head`, `MultiHeadAttention`, `FeedForward`, `Block`.
- `model.py` — GPT: embeddings + 3 Blocks (4 heads) + final LayerNorm + head.
- `train.py` — training loop, held-out val split, **saves best-val weights**.

**Study 2 (current)**
- `corpus2.py` — announcement/movement corpus. Verifies isolation, no-zeros,
  bridge coverage, token balance.
- `measure2.py` — probes movement magnitude, dose-response, correlation.
- `seeds2.py` — N-seed replication with seed-level t-tests.

**Study 1 (completed)**
- `corpus_min.py` — relational corpus with all fixes + verification.
- `measure.py` — condition battery.
- `seeds.py` — 5-seed replication.
- `chain.py` — relational-distance / combinatorial-entailment probe.
  **WRITTEN BUT NEVER RUN.**

## Pipeline
```
python corpus2.py    # -> corpus.txt (check isolation + no-zeros PASS)
python train.py      # -> model.pt (best-val), ~60s CPU
python measure2.py   # -> dose-response table
python seeds2.py     # -> the actual result, ~12 min
```
CPU only, no GPU needed. `pip install torch scipy`.

Study 1 and Study 2 both write `corpus.txt` — **delete it between studies** so a
stale corpus can't be trained on silently.

## NEXT STEPS
1. **Literature check** — still outstanding, still the gating question. Does
   cross-domain transformation of function in transformers exist in print? This
   decides whether Study 2 is a contribution or a replication.
2. **Fix the `ctrl` regime split** — replace the empty baseline with a
   fixed-mid-intensity announce-only symbol.
3. **More seeds** — slope is safe at p=.0002; secondary contrasts underpowered.
4. **Widen intensity range (1–12)** — tests whether compression is regression to
   the mean or a real ceiling.
5. **Mutual entailment** — `foo` announces more than `STIM_A`; does the reverse
   relation hold? Classic RFT criterion, never tested.
6. Only then: scale up.

## Person / context (for tone)
- Bosnia-based SDET with a psychology master's + behaviorism/RFT background,
  two regionally-published papers (one in Scopus/WoS-indexed Primenjena Psihologija).
- Learning LLMs from scratch via Karpathy videos — BUILD TOGETHER, piece by piece,
  each explained. Prefers small runnable pieces, inspect output, go slow.
- Goal: pragmatic progress toward better-paid AI-adjacent work (eval/testing/research).
  The experiment is a vehicle for understanding + a portfolio artifact.
