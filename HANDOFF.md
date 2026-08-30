# Handoff — start here

**Date: 2026-08-27. Deadline Nov 2, 2026 (67 days).**
You are picking up a Kaggle competition entry with no prior context. Read this
file fully before touching anything; several obvious-looking moves have already
been tested and rejected, and two of the traps here have each cost a day.

---

## 1. What this is

[ARC Prize 2026 (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).
Puzzles give a few input→output grid pairs, then a new input; you must produce
the exact output grid. Two attempts per output, either counts.

Our entry is a port of **NVARC**, the 2025 winner, running their public
Apache-2.0 checkpoint. No training of our own. Per puzzle it fine-tunes on that
puzzle's own examples (test-time fine-tuning), decodes it from 16 rotated and
recoloured views via a depth-first search over grids, re-scores the candidates
under 8 fixed augmentations, and submits the top 2.

**Current score: 29.31% / 27.64%** — same code, two runs. Rank ~602 of 1594.

## 2. The goal

Ken's target: **above 30% and a top 100–300 finish**, on the competition track.
Not the prize positions (8th is ~33.9%). The field is packed — roughly 770 teams
inside 6.7 points — so **+2pp is worth several hundred ranks.**

## 3. Do this first: Tier A

Full reasoning in the Notion page *arc-prize-next-steps-2026-08-26* (see §8).
Ordered by expected value. None of these costs a submission day.

### A1. Diff the public 33.89% notebooks against `submission/src`
Public notebooks running this same stack reportedly score **33.89 / 32.22 /
31.81%** while we sit at 29.31%. If real, that is a diff against readable code,
not research. A competitor publicly reported being stuck at 28–29% with settings
he believed identical (Kaggle `discussion/733930`), unanswered.

**Do this first, and re-verify the three numbers before trusting them** — they
were recorded 2026-08-21 and public notebooks get revised. This is the least
verified claim in the whole plan and it is load-bearing for the rest.

Diff, in priority order: the installed dependency set; per-puzzle and DFS time
caps; augmentation counts at fine-tuning, inference and re-scoring.

### A2. Sweep test-time fine-tuning — the only lever that touches the real bottleneck
**This is the most important item.** Everything previously tested was search or
selection, and all of it addresses a 7.6% problem. Test-time fine-tuning is the
only family that moves the 67.4% (see §4).

Our config is a faithful copy of NVARC's, which makes it *unswept*, not optimal:
`augment(n=16)`, `num_train_epochs=1`, `lr=5e-5`, `r=256`, `alpha=32`. Nobody
tuned it for the 4B or for the eval distribution. For contrast, a 200M LongT5
scoring 27.08% on this exact hardware used **35 epochs/task, r=8, lr 8e-4**.

Sweep as one bundle: epochs 1→2, `augment(n=)` 16→32, lr 5e-5→1e-4.

**It has to be paid for.** There is no idle budget (§5). Measure the fine-tuning
share of the 550s median first — it is already in the run logs — so the trade is
explicit rather than discovered when a run truncates.

Risk: more fine-tuning on 16 augmented copies of 3–5 demo pairs can overfit.
That is what the eval loop is for.

### A3. Conditional floor retry
Lowering the candidate probability floor **globally** was measured harmful
(§6). But that pays the cost on all 240 puzzles including the ones already
working. Instead: keep 0.2, and for puzzles that come back with an empty or
degenerate candidate set, retry the search at 0.05 with leftover budget. It only
fires where the puzzle currently scores zero, so it cannot regress anything.
`ARC_MIN_PROB` is already configurable.

### A4. Union TRM into `attempt_2` rather than voting on it
NVARC bolted on TRM (7M params, public weights `cpmpml/arc-prize-trm-031`) and
got nothing: 27.22 → 27.22. But their own words: *"about 2 or 3 puzzles solved by
TRM were not solved by Qwen3. Unfortunately, these were not always picked by
Qwen3 scoring."* That is a **selector** failure, and pass@2 sidesteps it —
reserve `attempt_2` for TRM's top grid when TRM produced something ours never
did. Union, not vote. Structurally cannot make things worse.

Caveat: TRM cost NVARC days of dependency conflicts. Keep it behind a flag.

### A5. Speculative decoding / fp8 KV cache
The ARChitects report **4.7× on DFS sampling** and 5.8× on scoring, on Kaggle
hardware. Do **not** spend a speedup on a wider search — the data says search
width is not the constraint. Spend it on A2 and A3.

---

## 4. The diagnosis, and why it dictates everything above

From `analyze_candidates.py` on the 120-task eval split, 4B checkpoint:

```
outputs analysed          : 172   (120 tasks; some have 2-3 test inputs)
correct grid generated    :  56 (32.6%)   the answer was in the candidate pool
never generated           : 116 (67.4%)   <- GENERATION gap
submitted correctly       :  43 (25.0%)   it ranked 1st or 2nd
generated but ranked out  :  13 ( 7.6%)   <- SELECTION gap
```

Two-thirds of the time **the model never proposes the right grid at all**, in any
of the 16 views. Ranking cannot recover those — the answer is not in the pool.

- Perfect ranking, same candidates → 32.6%. Worth +7.6pp, and the current ranker
  already captures most of it (37 of 56 at rank 1).
- The remaining 67.4% needs better generation: more/better test-time fine-tuning,
  better training data, or a better base model.

**Any work not aimed at generation is aimed at the small number.**

---

## 5. Measurement rules — read before running anything

**Noise floor is 1.67pp**, measured by resubmitting byte-identical code
(29.31 vs 27.64). A single submission cannot resolve anything under ~2pp.
Treat the leaderboard as confirmation, never as measurement.

**The eval split IS a valid offline yardstick — for the 4B.** It scores 26.1%
there against NVARC's published 25%, within 2.4pp of our leaderboard result.
Set `ARC_TASK_SET=eval` and a commit run gives a real number for zero
submissions. Use `ARC_TASK_LIMIT=60` to halve the cost when screening.

> Do **not** generalise from the 2B, which scores 82.6% on the same split.
> That checkpoint is contaminated. Earlier notes wrongly concluded from it that
> no public set could measure anything, and several submissions were wasted on
> questions a commit run could have answered.

**Commit runs are what the quota rations**, not submissions. A 120-task eval run
is ~5.2h wall and ~10h quota (L4×4 bills at 2×) against 30h/week — so about
**3 offline experiments per week, ~30 left**. Submission GPU time is not charged.

**Compare per-task sets, not aggregate rates.** On 120 tasks at p≈0.26 the
binomial standard deviation is ~4pp, so "26.1% → 29.4%" unpaired is meaningless.
The runs are paired: diff the solved-task sets and count flips. Roughly **8 net
flips** on 120 tasks before a difference is real.

**Batch orthogonal changes.** With ~30 experiments and a 1.67pp floor,
one-change-per-run loses. Bisect only when a bundle regresses.

---

## 6. Already tested and rejected — do not redo

| Lever | Verdict | How we know |
|---|---|---|
| Lower the probability floor globally | **Harmful** | 19.9h against an 11.6h budget; pass@2 flat at 28.0% across every floor below 0.2 |
| Raise the floor above 0.2 | **Harmful** | pass@2 falls monotonically: 26.4 → 25.2 → 24.5 → 23.9 |
| Length-normalised candidate scoring | Worse | 139 vs 144 of 151 |
| Dual-selector attempts | No gain | 144, unchanged, every pairing |
| Alternative rankers (probmul, hits, beam) | All worse | 143 / 130 / 127 vs 144 |
| Queue ordering by puzzle size | No gain | longest-first −1.3 solves, shortest-first +0.7 of 240 — under the noise floor |
| Adaptive per-puzzle time budget | Not worth it | truncation retains only ~34% of solve rate, so rationing loses more than it saves |
| `qwen3.patch` / KV-cache corruption | **Not a bug** | probe: batched vs batched 4/4, batched vs bsz=1 4/4; and `turbo_dfs` never rebinds `cache`, so siblings cannot interfere |

0.2 is the measured knee. NVARC chose it correctly.

---

## 7. Operational traps

**The notebook currently pushed is in `eval` mode.** Flip `ARC_TASK_SET` to
`"test"` before submitting. There is a guard that forces this on a scored rerun,
but do not rely on it.

**Every `kaggle kernels push` resets the accelerator** to the default GPU, a
P100 — and the image's torch 2.10 refuses to run on it at all (sm_60 below its
sm_70 minimum). **Re-set L4 ×4 in the editor after every push.** `machine_shape`
in the metadata is ignored; the API normalises whatever you send.

**`example/arc-agi_test_challenges.json` is not a sample of the eval set**,
despite Kaggle's Data tab. All 240 of its tasks are *training* tasks.
`validate_local.py` asserts this.

**Scored rerun logs are not retrievable.** Only committed-version output is. So
per-puzzle telemetry only helps for commit and eval runs.

**`kernels output` ignores the version argument** and serves one cached artifact.
Downloads are slow once `inference_outputs` exists (~1900 files) — background them.

**Artifacts live in `/tmp` and get cleaned.** The 2B baseline data was already
lost this way. Re-download rather than assuming.

**The margin is thin.** A 240-task run uses ~11:30 of an 11.6h budget, implying
~678s/puzzle — about 9% slower than eval. Simulation: ~1.7 tasks of 240 never
started (p90 7). Anything that slows a puzzle down eats directly into that.

**Silent success is the recurring failure mode.** Four times a run reported
COMPLETE while producing nothing useful. A shell command's exit code does not
fail a notebook cell. A guard that infers failure from a downstream artifact
breaks when anything upstream changes it — we broke our own guard exactly that
way, with a fix from a hardening list. Verify against the thing itself.

---

## 8. Where things are

```
PLAN.md                       status, corrections to earlier assumptions
submission/README.md          how the pipeline works, every tuning knob
submission/src/               source of truth; the notebook is a build artifact
submission/validate_local.py  28 offline checks, no GPU
submission/analyze_candidates.py   failure taxonomy + probability-floor sweep
NVARC/                        reference material from the 2025 winner + the paper
```

```bash
python3 submission/validate_local.py     # always, before pushing
python3 submission/build_notebook.py     # regenerate the notebook from src/
cd submission && kaggle kernels push     # -> wallfou/arc2-nvarc-2026
```

**Notion**, under Projects → ARC-Prize (ask Ken for access if the connector is
not set up):
- *arc-prize-next-steps-2026-08-26* — the Tier A/B/C plan this file summarises,
  with a corrections callout at the top. **Read the callout before acting.**
- *Iteration 1 : Notes*, *Iteration 2 : Notes* — full debugging history
- *Iteration 2 → Handoff to Iteration 3* — the long-form version of this file
- *Background Reading & Technical Landscape* — technique survey, reading list

---

## 9. The honest strategic picture

The recipe is near its ceiling around 30%, measured from several directions.
Reaching 8th place at ~33.9% needs something the published method does not
contain — better training data or a better base model, which is Tier B in the
plan and costs money and weeks.

**But Ken's stated goal is above 30% and top 100–300, which Tier A alone can
plausibly reach** if the public-fork delta is anywhere near real. Start there.

Also live, and not to be forgotten: the writeup award ($275k, judged on Theory
and Novelty as much as Accuracy) and the Paper Track ($450k, 150 participants
against 6,238 competition entrants, $375k of it a threshold prize not requiring
a high score). **Register for the Paper Track — ties break by earliest
timestamp.** Deadline Nov 8; artifacts open-sourced within 7 days, dual-licence
CC0 + MIT-0.
