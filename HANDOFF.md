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

### A2. Learning rate — ❌ NULL, measured 2026-09-06 (v47). Epochs/augment do not fit.

Swept 5e-5 / 1e-4 / 2e-4 **within one run**, 40 tasks x 3 arms, paired per task.

```
lr 5e-05   10.33/40 = 25.83%   <- baseline, best
lr 1e-04    9.83/40 = 24.58%
lr 2e-04    9.83/40 = 24.58%
```

**A 4x span moves 2 of 62 outputs.** 60 of 62 are identical across all three
arms, and the two that differ move in *opposite* directions — 1e-4 loses
`4c3d4a41_0`, 2e-4 loses `2d0172a1_0`, neither loses the other's. A systematic
lr effect would have the 2e-4 arm lose the 1e-4 output *and more*. This is the
signature of per-task decode nondeterminism, not of a learning-rate response.

Sanity check passed: the 5e-5 arm scores 25.83% on these 40 tasks against 26.11%
on the full 120, so the harness is behaving.

**The rest of A2 is not reachable.** `num_train_epochs` 1→2 and `augment(n=)`
16→32 each add ~3.2h to a 240-task run against ~1.2h of slack (§5), and TTFT is
30.2% of runtime. lr was the only free component and it is spent.

Honest limit: 2 discordant outputs means nothing under ~1.6pp (1 output of 62)
is resolvable. This is a clean null **for a large effect**, not proof of zero.

### A2 (original text, kept for the reasoning)
**This is the most important item.** Everything previously tested was search or
selection, and all of it addresses a 7.6% problem. Test-time fine-tuning is the
only family that moves the 67.4% (see §4).

Our config is a faithful copy of NVARC's, which makes it *unswept*, not optimal:
`augment(n=16)`, `num_train_epochs=1`, `lr=5e-5`, `r=256`, `alpha=32`. Nobody
tuned it for the 4B or for the eval distribution. For contrast, a 200M LongT5
scoring 27.08% on this exact hardware used **35 epochs/task, r=8, lr 8e-4**.

Sweep as one bundle: epochs 1→2, `augment(n=)` 16→32, lr 5e-5→1e-4.

**It has to be paid for.** There is no idle budget (§5) on a 240-task run.

> **ANSWERED 2026-08-30 (v40), and A2 as specified does not fit.**
> The phase breakdown was never in the logs despite this file and the Notion
> plan both saying so; `ttft_s`/`decode_s`/`retry_s` were added and have now run:
>
> ```
> total   22.84 GPU-h   685s/task
>   TTFT   6.91 GPU-h   30.2%   median 167s
>   decode 14.13 GPU-h  61.9%
>   retry   1.80 GPU-h   7.9%
> ```
>
> **Test-time fine-tuning is 30.2% of the run** (32.8% excluding the retry).
> So `num_train_epochs` 1→2 adds ~6.9 GPU-h per 120 tasks:
>
> | | 120-task eval | 240-task scored |
> |---|---|---|
> | today, no retry | 5.26h | **10.5h** (budget 11.67h) |
> | 2 epochs | 6.99h | **14.0h** — over by 2.3h |
>
> **It is measurable on eval and not deployable on a submission.** Measure it
> anyway — but know before you start that a win means finding 2.3h, and that
> the `augment(n=)` 16→32 half of the bundle costs decode time on top.

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

**Confirmed on our own run, independently.** v40 shipped with the retry on
before this was known. It fired on 20 of 120 tasks, re-decoded 154 batches
across 24 outputs, and cost 1.80 GPU-h (7.9% of the run):

```
outputs touched    24     de-starved 8     became correct 0
score on touched   3.00 -> 3.00            +0.00 tasks
```

**Exactly zero, matching his run exactly.** Two independent measurements, on
different code paths — his a separate re-fine-tuned pass at 24 origins, ours
inline on the existing adapter — both land on zero. It de-starves reliably and
converts nothing. `ARC_RETRY_STARVED=0`: 7.9% of a budget with 1.5% of slack,
for no flips.

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

> **But the eval loop has its own noise floor, and it is WORSE than the
> leaderboard's. Measured 2026-08-30, v40 against the v39 baseline: ±3.47pp.**
>
> v40 scored 29.58% against the baseline's 26.11% — a headline +3.47pp. Split by
> whether the retry *actually touched* the output, read out of the run log:
>
> ```
> TOUCHED   ( 24 outputs)  3.00 ->  3.00   +0.00 tasks = +0.00pp
> UNTOUCHED (148 outputs) 28.33 -> 32.50   +4.17 tasks = +3.47pp  <- ALL OF IT
> ```
>
> **Every one of the 7 gains and 3 losses is on an output the change never
> touched.** `PYTHONHASHSEED=0` is already set, so this is **intrinsic batched-
> search nondeterminism**, which NVARC documents (they measured batch-invariant
> kernels at 17% slower and dropped them).
>
> Partition on what the run log says was touched, **not** on what the baseline
> looked like. The first cut of this analysis split by baseline starvation and
> got +0.83pp for the retry — wrong, because the retry fires on the *new* run's
> starvation and nondeterminism moves which outputs those are.
>
> **So a single eval run cannot resolve anything under ~3pp either.** Two
> defences, and you need one of them:
> 1. **Partition the outputs by whether the change could reach them** and score
>    the halves separately, as above. This is what made v40 readable, and it is
>    free. Only works for changes with an inert subset (A3 had one; A2 does not).
> 2. For changes that touch everything, **sweep within a single run** — this
>    works, and it is the most useful thing built this iteration. Enqueue one
>    work item per (task, variant) and route each to its own output directory
>    (`ARC_LR_SWEEP` does exactly this; `starter.py` + `SWEEP_DIR`). Every pair
>    then shares a machine, a run and a clock, which removes the run-level
>    variance that dominates, and `make_submission` prints a score per arm so
>    the answer is in the run log rather than behind a ~1900-file download.
>
>    Measured effect on v47: **2 discordant outputs of 62 (3.2%) across a 4x lr
>    span**, against ~10 flips of 172 (5.8%, worth 3.47pp) between runs. Costs
>    one extra fine-tune+decode per arm, so N arms is N times the run. Enqueue
>    task-major so a truncated run leaves every arm covering the same prefix.
>
> **Where the noise most likely comes from — and it is not what the plan said.**
> The plan pointed at `madarshbb/reproduce-nvarc-2025-results` as "a
> batch-invariant build using `thinking-machines-lab/batch_invariant_ops`".
> Pulled it 2026-09-05: **it contains no batch-invariant code at all** — zero
> hits for `batch_invariant`, `deterministic`, or `cublas`. It is a plain NVARC
> reproduction with our exact knobs. Third misattributed claim from that page.
>
> The likelier dominant source is **wall-clock truncation**. `DFS_BUDGET_S=540`
> and `PUZZLE_BUDGET_S=1200` cut the search on elapsed time, and **13 of 120
> tasks hit the 1200s cap** on the baseline run. Wall clock varies between runs
> with GPU contention and scheduling, so a capped task decodes a different
> number of views each time — different candidates, different score. Batch-
> invariant kernels would not fix that; they address a different problem
> (results changing with batch *composition*, which is fixed here by seed).
>
> **Untested hypothesis, cheap to check**: on the next paired eval runs, see
> whether flipped outputs are disproportionately on tasks at/near the cap. If
> so, the fix for *measurement* runs is deterministic truncation — cap on work
> done (batches decoded, nodes expanded) rather than on `time.time()`. No new
> dependency, and it makes A/B runs comparable. Keep wall-clock caps for scoring
> runs, where finishing inside 12h is what matters.

> Do **not** generalise from the 2B, which scores 82.6% on the same split.
> That checkpoint is contaminated. Earlier notes wrongly concluded from it that
> no public set could measure anything, and several submissions were wasted on
> questions a commit run could have answered.

**Commit runs are what the quota rations**, not submissions. A 120-task eval run
is ~5.2h wall and ~10h quota (L4×4 bills at 2×) against 30h/week — so about
**3 offline experiments per week, ~30 left**. Submission GPU time is not charged.

**Submission count predicts score, and we are badly under-sampled.**

Sharpest version, 2026-09-05 — teams running *this exact NVARC stack*:

| team | rank | score | subs |
|---|---|---|---|
| `koushikrudra` | 9 | 33.89 | 110 |
| `yusuketogashi` | 159 | 31.81 | 42 |
| **`madarshbb`** | **200** | **31.39** | **107** |
| `boristown` | 651 | 29.72 | 31 |
| **us** | **731** | **29.31** | **6** |
| `mikelou1` | 827 | 28.61 | 11 |

`madarshbb`'s public notebook has **byte-identical solver knobs to ours** — 540s
DFS, 1200s task, 1 epoch, lr 5e-5, floor 0.2, `augment(n=16)`, `n=2` views — and
scores +2.08pp with 107 submissions. Every team above us has 30+; we have 6.

**The leaderboard scores all 240 tasks.** The smallest gap between distinct
scores is 0.1389pp = 1/3 of a task out of 240, so there is no held-out public
subset: a good run's score is locked in, not regressed at the end. That is what
makes resubmission worth anything at all.

Modelled on our three same-config runs (29.31 / 27.64 / 29.31, mean 28.75):
at sd 1.2, ~48% of touching 30.97 within 20 submissions and ~73% within 40.
Highly sensitive to sd, which n=3 barely determines. **This harvests noise; it
does not improve the solution.** Top 100 (32.22) is out of reach this way.

Across all 964 teams scoring above 15%:

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

## 5a. The two-mode score, and why it does not need fixing

**Four scored runs of identical code gave exactly two values** — 29.31 / 27.64 /
29.31 / 27.64, differing by *exactly* 4.00 tasks (12 quanta of 1/3 task of 240).
Under a continuous-noise model, 4 draws landing in 2 buckets is ~1% likely. The
variance is **structural, two modes — not a spread.**

Leading explanation is the clock: a 240-task run needs ~11.34h of an 11.67h
window, so a ~4% slow run leaves ~15 tasks unstarted, which at a ~26% solve rate
is the 4 tasks observed. Eval cannot show this: 120 tasks use 5.2h of the same
window.

**But this is a tax, not a recoverable loss, and it was nearly a trap.**
Kaggle displays your *best* submission, and 29.31 is already on the board. If
the two modes are "finished the queue" and "did not", then **29.31 is the
un-truncated score and we are already showing it.** Fixing truncation would move
the leaderboard only if the *fast* runs also drop tasks — possible, unmeasurable
from here, and small. Chasing the 1.67pp buys nothing; raising the ceiling does.

**This also kills best-of-N resubmission** (§5): if the distribution is two
values, no number of submissions reaches 30.97. That strategy was recommended on
2026-09-05 from a Gaussian fit to 3 points and was falsified the next day.

### v45 — the squeezed-budget test, and what it actually settled

60 eval tasks into a ~2.26h window, meant to be 1.15x oversubscribed.
**It missed: calibrated with the 120-task mean (624s/task) against a 60-task
subset that runs at 538s/task, so it landed at 0.99x and just fit.** Result:
60/60 done, `queue_left=0`, `out_of_time=False`, 3 tasks hit the 1200s cap.
Truncation was not reproduced, so the hypothesis stands **untested**. To redo it
properly the window is ~1.95h (`ARC_TOTAL_BUDGET_S=7920`).

It did close both headroom levers, which is why it was still worth the quota:

| Lever | Verdict |
|---|---|
| `use_cache=False` in `calc_scores` | **No measurable gain.** Apparent -13.8% mean speedup is the cheaper subset. TTFT is the control — it *cannot* be affected by the change, and it moved -16.1% while decode moved -14.1%, so decode/TTFT actually rose 2.4%. Kept: free and algebraically identical, but do not count on it |
| Gradient checkpointing off (~1h) | **Unsafe.** `peak_train_mb` is 13129 MB and *identical on all 60 tasks* — with checkpointing on, activations are negligible and the peak is weights+optimizer. Off, activations scale with sequence length; at `max_seq_length=8192` that is >20 GB on top of 13.1 GB against a ~22.5 GB card. Long tasks would OOM, and the worker survives an OOM by *skipping the task*, so it would silently drop the biggest puzzles |

> **Lesson:** calibrate an experiment against the population it will actually
> run on. The squeeze used a mean measured on 120 tasks to size a window for a
> 60-task subset of them, and the 13% error was exactly enough to miss.
> Second lesson, more useful: **TTFT is a free internal control for any change
> that touches only decoding.** It scales with task size but cannot be affected
> by decode-side edits, so the decode/TTFT ratio removes the subset effect.
> Without it, v45 would have been written up as a 13.8% speedup that is not real.

---

## 6. Already tested and rejected — do not redo

| Lever | Verdict | How we know |
|---|---|---|
| Lower the probability floor globally | **Harmful** | 19.9h against an 11.6h budget; pass@2 flat at 28.0% across every floor below 0.2 |
| Lower the floor *conditionally*, on starved outputs only | **Worthless** | 43/172 with and without, zero net flips, on the published run of a strictly more aggressive version (A3). Recovers missing candidates; none is correct |
| `use_cache=False` in teacher-forced scoring | No gain | decode/TTFT ratio rose 2.4% (§5a). Harmless, kept |
| Gradient checkpointing off, to buy ~1h | **Unsafe** | training peak 13.1 GB is activation-free *because* checkpointing is on; off, 8192-token tasks blow a 22.5 GB card and get silently skipped (§5a) |
| Best-of-N resubmission | **Dead** | the score has two modes, not a spread; 29.31 is a ceiling and is already displayed (§5a) |
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
lost this way, and on 2026-09-05 the 4B baseline *and* v40 candidate files went
the same way — six days after download, directories intact but empty. The
derived numbers survive in this file and in commit messages; the raw candidates
do not. **If you download a run you intend to compare against later, copy it
somewhere durable**, or plan to re-download. Note `kernels output` serves only
the newest version's artifact, so once a newer version runs, the older run's
output may be unrecoverable.

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
