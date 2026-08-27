# Testing a prediction from Relational Frame Theory on a small transformer

Relational Frame Theory makes a specific claim about learning: if a symbol is
related to others in some way, it should acquire the *functions* of those others
without ever being directly trained on them. Tell someone an unfamiliar animal
is bigger than a bear, and they will back away from it — despite never having
seen it do anything.

That claim is hard to test in people, because you can never audit a person's
entire learning history. You can audit a model's.

So I built a 167k-parameter GPT from scratch and controlled its entire training
corpus — every unit generated from a spec I wrote, with checks that verify
exactly what each symbol was and was not exposed to on every build. Then I ran
two studies.

**Study 1 stated the relation explicitly and produced a null.**
**Study 2 stated it as a shared dimension and confirmed the prediction.**
The null is the reason Study 2 exists, and it is explained rather than buried.

Everything runs on CPU. No GPU. One command reproduces the full result in about
fifteen minutes.

```bash
pip install -r requirements.txt
python seeds2.py        # Study 2, 6 replications
```

---

# Study 1 — the explicit relation

## Design

The world is an agent (`AG`) and its distance from a stimulus, written in `#`
marks. The agent flees, approaches, or holds:

```
AG # # # STIM_A a04 AG # # # # # STIM_A       agent fled: gap grew 3 → 5
```

Two stimuli were grounded by direct training: `STIM_A` (agent flees) and
`STIM_T` (agent approaches). Then relations were taught over 120 throwaway
symbols, using `=` for "same behaviour" and `!=` for "opposite":

```
r083 != r094 | AG # # # r083 a05 AG # # # # r083 | AG # # # r094 t00 AG # r094
```

No symbol had a fixed direction — `r083` flees in one unit and approaches in
another — so the operator was the only reliable predictor. The model could not
memorise symbols; it had to learn what `=` and `!=` *do*.

Finally, the test statement, repeated 40 times, with the behaviour withheld:

```
foo = STIM_A |
```

`foo` was never shown moving. Not once. Verified, not assumed.

## Result

**With the statement in the prompt, it worked** — and it passed both controls
that a sceptic would demand:

| probe | moved | |
|---|---|---|
| `foo = STIM_A` | **+1.96** | flees, like its anchor |
| `foo != STIM_A` | **−1.32** | **opposition inverts** |
| `foo = STIM_T` | **−2.00** | **tracks which anchor** |

Inversion rules out simple association. Anchor-tracking rules out generic
priming. Replicated across 5 seeds.

**With the statement removed, nothing.** `foo` −0.31 against an unlinked control
at −0.13. Indistinguishable, despite 40 training exposures to the relation.

## Explanation

This is not a limitation of the architecture. It is a property of the training
objective, and it is derivable in advance.

The loss on `foo = STIM_A |` asks three things: predict `=` after `foo`, predict
`STIM_A` after `=`, predict `|` after that. **No term in that objective
references movement.** So the gradient with respect to *how far the agent moves
near `foo`* is exactly zero — not small, zero. Forty exposures or forty
thousand, it cannot write a behavioural function into a symbol the objective
never asks to behave.

There is a corollary worth stating, because it closes off the obvious fix: any
procedure that writes a movement function into `foo` requires training on `foo`
moving, and training on `foo` moving *is* grounding it. So durable transfer and
derivation-from-a-bare-statement are mutually exclusive by construction.

The way out is not more exposure. It is to ground the test symbol in a *different
response domain* — real training, real gradient, just not the domain being
tested.

---

# Study 2 — the shared dimension

## Design

Two worlds in one corpus.

**Announcements.** A symbol emits a number of `!`:

```
STIM_A ! !              announces 2
foo    ! ! ! !          announces 4
bar    ! ! ! ! ! !      announces 6
```

**Movement.** The agent flees a stimulus, as before. Magnitude 1 to 6.

```
AG # # p058 a02 AG # # # # # p058
```

**The bridge**, which is the load-bearing part. Nothing intrinsically connects
"announces 4" to "flees 4" — that link has to be taught, on *other* symbols. 102
bridge symbols each have an intensity and appear in **both** worlds: announcing
that many `!`, and fleeing that many `#`. That is what makes announcement-count
and movement-magnitude the same underlying dimension.

`foo`, `bar` and six probe symbols `q1`–`q6` appear **only in announcements**.
They never move. `ctrl` appears once as a bare token, with no announcements and
no movement — the floor.

There are **no zeros anywhere**. No neutral stimulus, no hold behaviour, every
movement at least 1. "Stay put" is not a trained response, so the agent always
moves and cannot decline to reveal what it encodes.

Then I ask what the agent does near a symbol that has never moved.

## Result

`foo` announced 4 and fled **4.29**. Across the six probe symbols the
dose-response is **monotonic at every level**:

| announced | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **moved** | 1.98 | 3.43 | 3.65 | 3.92 | 5.07 | 5.18 |

Slope **0.61**, p = **.0001**, positive in all six runs. Slope 1.0 would be
perfect transfer, so roughly 60% of the magnitude carried across. Mean
within-run correlation r = **.81** (SD .07).

The prediction could have failed four ways, and each has a control:

| if this were true | you would see | actually observed |
|---|---|---|
| training failed | anchor off 2.0 | **2.00 ± 0.00** |
| the model just moves randomly | no relationship to announcement | **slope 0.61, p=.0001** |
| any symbol moves this much | `ctrl` matches `foo` | `ctrl` 2.64 vs `foo` 4.29 |
| ordering not preserved | non-monotonic curve | **monotonic, all 6 levels** |

<details>
<summary>Full output of <code>python seeds2.py</code></summary>

```
GATE — is the anchor trained correctly? (announces 2, IS in movement training)
  STIM_A moved 2.00 ± 0.00  (want ~2.0)

TRANSFER — all of these NEVER appear in a movement unit
  test                       mean      sd        t        p      d   per-seed
  slope (announced->moved)   0.61 ±  0.14   t= 10.55  p=0.0001  d= 4.31   +0.42 +0.76 +0.45 +0.74 +0.62 +0.63
  foo - ctrl                 1.66 ±  2.04   t=  1.99  p=0.0516  d= 0.81   +3.35 -0.46 +3.93 -0.62 +3.10 +0.64
  bar - ctrl                 2.36 ±  2.29   t=  2.52  p=0.0265  d= 1.03   +4.95 -0.29 +5.00 +0.12 +1.70 +2.71
  bar - foo  (ordering)      0.71 ±  1.23   t=  1.41  p=0.1087  d= 0.58   +1.60 +0.17 +1.07 +0.73 -1.40 +2.07
  r (announced vs moved)     0.81 ±  0.07                                 +0.90 +0.88 +0.76 +0.75 +0.75 +0.85

DOSE-RESPONSE — mean movement per announced intensity
   announced    moved     sd
           1     1.98   0.89
           2     3.43   0.93
           3     3.65   1.15
           4     3.92   1.22
           5     5.07   0.35
           6     5.18   0.52

RAW MEANS
  STIM_A     2.00 ± 0.00      foo 4.29 ± 1.00
  bar        5.00 ± 0.93      ctrl 2.64 ± 1.57
```
</details>

## Explanation

The ordering transferred almost perfectly. The magnitude transferred partially.

Compression is visible at both ends — announced 1 moved 1.98 when it should have
moved 1; announced 6 moved 5.18 when it should have moved 6. Everything is
pulled toward roughly 3.5, the mean of the trained movement distribution. That
is why the slope is 0.61 rather than 1.0, and why the middle of the curve has
the highest seed-to-seed variance (sd 1.15 and 1.22 at announced 3 and 4, against
0.35 and 0.52 at 5 and 6): where two outcomes are nearly equally good, which one
a run picks is close to arbitrary.

Whether that compression is regression to the mean or a genuine ceiling is open.
Widening the intensity range to 1–12 would separate them — under regression the
curve stretches, under a ceiling it flattens above 6.

The mechanism is probably the one behind `king − man + woman ≈ queen`: intensity
becomes a direction in embedding space. What is different here is that it is
measured **behaviourally**, on a corpus where every exposure is auditable. I
never touched the embeddings — I gave the model a situation and recorded what it
did.

**Statistical note.** The unit of analysis is the **seed**, not the generation.
Pooling all ~600 generations per run and regressing would give a far smaller
p-value and would be wrong: generations from the same symbol on the same model
are 160 samples of *one thing*, not 160 independent observations. So n = 6, and
the p-values are honest but low-powered.

---

## Where the controls came from

Study 2's corpus generator runs six verification checks on every build. Each one
exists because an earlier version of Study 1 failed in a way I had not
anticipated. None of these bugs are in the results above — they are the reason
the current design is checkable at all. Every one was caught by a control, not by
reading the code.

- **My first run looked like a clean success.** It was not: the *unlinked control
  symbol* moved too. A symbol connected to nothing cannot produce direction, so
  the effect had to be coming from somewhere else. It was — I had accidentally
  made the agent's starting position predict which way it moved.
  → *now verified: one shared start range for all directions*

- **A "strong" effect of +5.24 was mostly context.** A control that put the anchor
  token in the prompt *without any relation* produced +3.16 of it on its own.
  → *now avoided: Study 2 probes carry no context at all, just `AG ## foo`*

- **A model with an excellent validation loss had learned nothing relevant.** `#`
  was 75.5% of all tokens, so cross-entropy was almost entirely measuring "can
  you count." The tokens that mattered were 1.6% — ignoring them cost the
  optimiser nothing, and it did. Found by printing a token histogram; fixing the
  encoding moved the effect from **−0.38 to +7.07**.
  → *now verified: the token histogram prints on every corpus build*

- **Discarded samples were biased.** Generations that ran past the token limit
  were dropped — and those were disproportionately the *largest* movements,
  quietly biasing against the very effect being measured.
  → *now reported: invalid-sample counts, per condition*

## What's in here

Built from scratch and commented throughout:

- `attention.py` — single head → multi-head → feed-forward → transformer block
- `model.py` — embeddings + 3 blocks + output head
- `train.py` — training loop, held-out validation, keeps best-val weights
- `corpus2.py` / `measure2.py` / `seeds2.py` — Study 2
- `corpus_min.py` / `measure.py` / `seeds.py` — Study 1
- `PROJECT_STATE.md` — full working notes, open questions, next steps

## Honest notes

- Transfer is partial: slope 0.61, not 1.0.
- Movement' is a description of the encoding, not a claim about representation.
  What is measured is a graded response class under symbolic control.
- The single-symbol contrasts against `ctrl` are **underpowered and unstable**.
  Per-seed values split into two regimes (+3.35, −0.46, +3.93, −0.62, +3.10,
  +0.64). The cause looks like `ctrl` itself — with no information at all, its
  behaviour is arbitrary and swings by seed (SD 1.57, against `STIM_A`'s 0.00).
  The dose-response avoids this by averaging six symbols and never referencing
  `ctrl`. Replacing the empty baseline with a fixed-mid-intensity symbol is the
  next change.
- That even/odd seed split **reproduced on a second machine** with a different
  PyTorch build, so it is systematic rather than noise. Currently unexplained.
- One architecture, one scale, n = 6 seeds.
- Connectionist modelling of stimulus relations is an established literature
  (RELNET; Tovar & Chavez 2012; Vernucio & Debert 2016). What is new here is the
  transformer, and a graded *behavioural* dependent variable rather than
  matching accuracy.
- Implementation was AI-assisted. The experimental design, the decisions about
  what did and did not constitute a valid test, and the diagnosis of each
  confound above are mine.
