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

ward better-paid AI-adjacent work (eval/testing/research).
  The experiment is a vehicle for understanding + a portfolio artifact.
