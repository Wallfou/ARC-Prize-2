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

> **Re-measured 2026-08-30 against the live leaderboard (1,712 teams). 30% no
> longer buys the stated rank — the field moved under us.**
>
> | Rank | Score today |
> |---|---|
> | 8 (last paying) | 34.44 |
> | 100 | **31.94** |
> | 200 | 31.39 |
> | 300 | **30.97** |
> | 500 | 30.14 |
> | **us, 29.31** | **~642** |
>
> In numbers the goal is **31.0% for top 300, 32.0% for top 100** — +1.7pp and
> +2.7pp from here. Against a 1.67pp noise floor, aim at **32%**, so that a
> bad-noise run still lands inside 300. "Above 30%" on its own is rank ~549.
>
> Refresh before planning; this moved ~2pp in four days:
> `kaggle competitions leaderboard -c arc-prize-2026-arc-agi-2 --download`


## 3. Do this first: Tier A

Full reasoning in the Notion page *arc-prize-next-steps-2026-08-26* (see §8).
Ordered by expected value. None of these costs a submission day.

### A1. Diff the public notebooks — ✅ DONE 2026-08-30. The premise was wrong; the answer is A3.

**The three numbers were misattributed.** Verified against the live leaderboard
by matching notebook authors to their teams:

| Author | Notebook title | **Actual LB** | Rank |
|---|---|---|---|
| `mikelou1` | "ARC AGI2 **LB33.89** Minimal Perfpatch" | **28.61** | 758 |
| `boristown` | byte-identical fork of the above | 29.72 | 596 |
| `koushikrudra` | "ARC AGI2 Original KG" | **33.89** | 9 |
| `yusuketogashi` | "ARC Baseline Rebuild" | **31.81** | 147 |
| `madarshbb` | "Reproduce NVARC 2025 Results" | 31.39 | 178 |
| us | — | 29.31 | 677 |

"LB33.89" is a *title*, not mikelou1's score — he is at **28.61, below us**, and
his is the notebook the old plan pointed at. And koushikrudra, cited here as the
competitor "stuck at 28–29%", is actually **rank 9** with 107 submissions; his
33.89 comes from a private notebook, not the public one bearing his name.
**There was no free 3–4pp sitting in the notebook the plan named.**

What the "Minimal Perfpatch" actually is: keep DFS logits on GPU and gather only
the ARC tokens rather than `.cpu().log_softmax(-1)` over the vocabulary, plus
`use_cache=False` in teacher-forced scoring. Algebraically identical — and its
stated rationale does not apply to us, because **this checkpoint's vocabulary is
16 tokens**, so "instead of every Qwen vocabulary logit" means 16 values, not
150k. Only the `use_cache=False` half is a real saving. Not worth a run alone.

Also checked: every public notebook uses `sorokin/pip-install-unsloth-flash-patch`
as its wheelhouse where we use `wallfou/arc2-wheelhouse`. Worth a look if you
ever need flash-attn, but `qwen3.patch` is already ruled out as the gap (§6).

`yusuketogashi/arc-baseline-rebuild` (31.81%, rank 147) is the one notebook
verified above us, and its edge looked like a conditional deep retry. **It is
not: that retry is measured worth exactly zero on its own published run (A3).**

What is left is the finding that matters more. **His primary pass scores 43/172
on the eval split — identical to ours, output for output** — and his deep pass
adds nothing. So there is *no code difference* behind his +2.5pp. What differs
is **38 submissions against our 6**, into a 1.67pp noise floor. See §5.

### A2. Sweep test-time fine-tuning — the only lever that touches the real bottleneck
**This is the most important item.** Everything previously tested was search or
selection, and all of it addresses a 7.6% problem. Test-time fine-tuning is the
only family that moves the 67.4% (see §4).

Our config is a faithful copy of NVARC's, which makes it *unswept*, not optimal:
`augment(n=16)`, `num_train_epochs=1`, `lr=5e-5`, `r=256`, `alpha=32`. Nobody
tuned it for the 4B or for the eval distribution. For contrast, a 200M LongT5
scoring 27.08% on this exact hardware used **35 epochs/task, r=8, lr 8e-4**.

Sweep as one bundle: epochs 1→2, `augment(n=)` 16→32, lr 5e-5→1e-4.

**It has to be paid for.** There is no idle budget (§5) on a 240-task run.

> **Correction 2026-08-30: the fine-tuning share is _not_ in the run logs.**
> Both this file and the Notion plan said it was. The logs record only
> `seconds` per puzzle (fields: `key ok seconds peak_infer_mb done avg_seconds
> affordable outputs task_set`) — there is no TTFT/DFS/scoring breakdown, so
> Open Question 5 is unanswered and needs instrumentation added first.
> Measured totals that _are_ available: median 550s, mean 624s, p90 1217s,
> 120 eval tasks = 5.20h wall = 20.8 GPU-h, projecting to ~10.4h for 240.

Risk: more fine-tuning on 16 augmented copies of 3–5 demo pairs can overfit.
That is what the eval loop is for.

### A3. Conditional deep retry — ❌ MEASURED WORTHLESS 2026-08-30. Do not redo.

Killed by the strongest evidence available: **the other team's own published run
output.** `yusuketogashi/arc-baseline-rebuild` (31.81% LB, rank 147) ships both
its primary-only and its deep-augmented submission for the same 120-task eval
run. `kaggle kernels output yusuketogashi/arc-baseline-rebuild` retrieves them.

Scored against `example/arc-agi_evaluation_solutions.json`:

```
primary only : 43/172 (25.00%)
deep applied : 43/172 (25.00%)
gained 0 | lost 0 | NET FLIPS +0
```

From a deep pass **strictly more aggressive than anything we would build**:
floor 0.08 *and* 24 inference origins *and* 16 scoring views *and* a 2400s task
cap, 1.9h over 18 selected tasks, `deep_status=COMPLETED`, return code 0.

It worked mechanically and still scored nothing: outputs with no candidate at
all fell **10 → 4**, and 13 submitted answers changed — **none of the recovered
grids was correct.** The retry converts "no answer" into "wrong answer".

**The lesson generalises past this lever.** A starved output is not the floor
pruning a grid the model knew; it is the model not knowing the grid. That is
§4's generation bottleneck restated, and it means *no search or floor setting
fixes it.* Treat "spend more search on the hard ones" as closed.

Code is kept and defaults **off** (`ARC_RETRY_STARVED=0`): `decode_pass` in
`arc_solver.py`, knobs in `arc_config.py`, census in `analyze_starvation.py`.
Re-enable only if the *generator* improves enough to change the premise.


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

**Submission count predicts score, and we are badly under-sampled.** Across the
964 teams scoring above 15%:

| submissions | teams | median score |
|---|---|---|
| 1–5 | 484 | 29.38 |
| 6–15 | 259 | 30.56 |
| 16–40 | 146 | 31.11 |
| 41–80 | 53 | **32.22** |
| 81+ | 22 | **32.36** |

Correlation +0.33. **We have 6 submissions and sit at 29.31.** The confound is
real — teams that submit more also iterate more — but the cleanest control
available says selection is a large part of it: `yusuketogashi` scores *exactly
what we score* on eval (43/172) with a deep pass worth zero, and is +2.5pp ahead
on the leaderboard with 38 submissions. Best-of-N against a 1.67pp floor.

Submissions do **not** consume the GPU quota (only commit runs do), and the cap
is 1/day with ~63 days left. Banking repeat runs of the known-good config is
close to free and buys ranks directly. Two submissions are chosen for final
judging, so keep the best one selected at all times.

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
| Lower the floor *conditionally*, on starved outputs only | **Worthless** | 43/172 with and without, zero net flips, on the published run of a strictly more aggressive version (A3). Recovers missing candidates; none is correct |
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
submission/analyze_starvation.py   starved-output census; sizes the A3 retry
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
