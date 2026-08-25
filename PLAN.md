# ARC Prize 2026 — Iteration Plan

> **Status 2026-08-25 — leaderboard 29.31% / 27.64% (same config twice), 69 days left.**
>
> **Two earlier conclusions were wrong. Both corrected below.**
>
> **1. The public eval split IS a usable yardstick — for the 4B.** Our 4B scores
> 26.1% on it against NVARC's published 25% for the same checkpoint, and within
> 2.4pp of our own leaderboard result. The 82.6% that led us to call eval
> "saturated and useless" came from the **2B**, which is the anomaly: NVARC also
> report 25/120 for their 2B against our 99/120, so `sft141` appears to have the
> eval puzzles in training in a way `sft139` does not.
> **A commit run (~5.2h wall, ~10h quota; halve it with `ARC_TASK_LIMIT=60`) now
> replaces a submission for A/B work.** This is the first measurement loop that
> costs less than a day.
>
> **2. The bottleneck is generation, overwhelmingly.** Measured on the 4B, not
> the 2B:
>
> | | 2B (contaminated) | 4B (what we run) |
> |---|---|---|
> | Correct grid generated | 87.8% | **32.6%** |
> | Never generated | 12.2% | **67.4%** |
> | Generated but ranked out | 4.1% | 7.6% |
>
> The model cannot produce the right grid two-thirds of the time. Not a search
> problem, not a ranking problem — capability and training data.
>
> **Measured noise floor: 1.67pp on byte-identical code** (29.31 vs 27.64). A
> single submission cannot resolve anything under ~2pp.
>
> **Every cheap lever is exhausted:**
>
> | Lever | Verdict | Cost to learn |
> |---|---|---|
> | Lower the probability floor | Harmful — 19.9h against 11.6h, pass@2 flat | 1 commit run |
> | Raise the probability floor | Harmful — pass@2 falls monotonically above 0.2 | free, offline |
> | Length-normalised scoring | Worse (139 vs 144 of 151) | free, offline |
> | Dual-selector attempts | No gain | free, offline |
> | Four alternative rankers | All worse than the current one | free, offline |
>
> 0.2 is the measured knee; NVARC chose it correctly. Selection is worth at most
> 7.6pp and the current ranker captures most of it.
>
> **Also corrected: there is ~2.4h of idle compute, not 4–8h.** Median 550s per
> puzzle on real eval tasks projects to 9.2h of the 11.6h budget. The earlier
> figure came from four easy smoke puzzles.
>
> **Next.** Closing a 67.4% generation gap means better training data or a better
> base model — the expensive lever, now the only one left, and finally testable
> against a 26.1% offline baseline. In parallel, the writeup award (\$275k) and
> Paper Track (\$450k, 150 participants vs 6,238 entrants) remain the better
> expected value, and this iteration's material is strong: a measured noise floor,
> five rejected levers, and two of our own conclusions overturned by measurement.

---

> Written 2026-08-06, when the NVARC port was finished and validated.
> Companions: the Notion page *Background Reading & Technical Landscape*
> (technique survey, reading list) and [submission/README.md](submission/README.md)
> (how the port works).
>
> This supersedes the strategy section of the old `research.md`, written in June
> before the 2026 leaderboard was known. That doc now lives in Notion, with its
> wrong calls marked.

---

## 1. Where things stand

**88 days to the final submission deadline** (Nov 2, 2026). At one submission per
day that is **at most 88 more scored runs**, and in practice far fewer, because a
4B run consumes 12 hours of wall clock.

| Milestone | Date | Days |
|---|---|---|
| Entry + team merger deadline | Oct 26, 2026 | 81 |
| **Final submission** | **Nov 2, 2026** | **88** |
| Paper Track | Nov 9, 2026 ⚠️ | 95 |
| Writeup artifacts due (7 days post-deadline) | Nov 9, 2026 | 95 |
| Winners announced | Dec 4, 2026 | 120 |

⚠️ arcprize.org says papers are due Nov 8; the Kaggle API says Nov 9. Assume Nov 8.

**Already entered:** `arc-prize-2026-arc-agi-2` and `arc-prize-2026-paper-track`
(both `userHasEntered=True`). Not entered: arc-agi-3.

**Prize structure — $700K, and the score is not the only path to it:**

- **$275K Progress Prizes** by leaderboard rank, top 8. 1st $75K, 8th $15K.
- **$275K Grand Prize** (called "Innovation Prize" in the binding rules) — awarded
  to the best **Solution Writeup**, not the best score. Six equally weighted
  criteria: Accuracy, Universality, Progress, Theory, Completeness, Novelty.
- **$150K Bonus** — unlocked only if someone scores **≥85%**. Nobody is close.
- Separate **$450K Paper Track**, already entered. Papers must link to a Kaggle
  submission but need not score well.

**The gap.** Public leaderboard as of 2026-08-06:

| # | Team | Score |
|---|---|---|
| 1 | nvbanana (very likely Puget, i.e. NVARC again) | **67.50** |
| 2 | rabbithole | 47.78 |
| 3 | Junhua Yang | 37.22 |
| 8 | last paid position | ~33.89 |

NVARC won 2025 with 24.03% private / 27.64% public. **A faithful recreation lands
around 10th–15th — outside the eight prize positions.** The bar roughly tripled
in a year.

**The encouraging read:** compute is fixed at L4×4 / 12h / no internet for
everyone, unchanged from 2025. So 67.5% cannot come from more compute. It comes
from a better model or better data — which is, in principle, reproducible.

---

## 2. What the port gets us

[submission/](submission) is a validated recreation of NVARC. Zero training
required: the fine-tuned weights are public and Apache 2.0.

| Model | Runtime | 2025 public LB |
|---|---|---|
| `sorokin/qwen3_2b_grids15_sft141` | 6h10m | 22.22% |
| `sorokin/qwen3_4b_grids15_sft139` | 12h | 29.72% |

Its value is **infrastructure, not score**: a proven 12h/L4×4 harness with
per-task TTFT, batched DFS, augmentation re-scoring, and a submission writer that
cannot silently emit garbage. Every idea below plugs into it.

---

## 3. Phase 0 — ship the baseline (this week)

Goal: a real leaderboard number and a known-good harness.

1. **2B commit run.** 4 tasks, ~30 min. Only checks that the install path and the
   Unsloth patch survive the Aug 2026 Kaggle image. *The most likely failure in
   the whole plan is `qwen3.patch` no longer applying — it is pinned to
   `unsloth==2025.9.7`, a Dec 2025 release.*
2. **2B full submission.** ~6h. Expect ~22%. Record: wall clock per task, the
   pace log, peak GPU memory, and how many tasks actually completed.
3. **4B submission.** ~12h. Expect ~30%. This is the honest ceiling of pure
   recreation.

**Exit criteria:** a leaderboard score within ~2pp of 29.72% (paper §3.5 reports
1–2pp run-to-run variance), and a measured per-task time budget.

If the 4B lands near 30%, the recreation is *done* and correct. Everything after
this is about the 37pp gap.

---

## 4. Phase 1 — measure before optimising (~1 week)

Cheap, no GPU rental, high information value.

- **Build a real local validation loop.** Score on the 120-task public eval split.
  Never on `arc-agi_test_challenges.json` — all 240 of those tasks are *training*
  tasks (verified; `validate_local.py` asserts it).
- **Correct for contamination.** NVARC seeded 55% of their SFT data from ARC-AGI-2
  public *eval* descriptions — 29 hand-labelled, 91 LLM-summarised (paper §2.1,
  §3.5). The released checkpoints have semantic exposure to exactly the 120 tasks
  you would validate on. Treat local-vs-LB gap as uninformative until you have a
  held-out split the checkpoint never saw.
- **Per-task failure taxonomy.** For each eval task: did DFS produce the correct
  grid *anywhere* in its candidate set but rank it below 2nd? That distinguishes
  a **generation** problem from a **selection** problem, and they have completely
  different fixes. NVARC's `benchmark_selection_algos()` already prints the raw
  material for this.
- **Where does the time go?** TTFT vs DFS vs re-scoring, per task. Determines
  whether buying more compute per task is even possible.

**Deliverable:** a one-page diagnosis of *why* the model fails, not just how often.

---

## 5. Phase 2 — close the gap

Ranked by expected value per unit of effort and cost. Do them in order; each is
independently submittable.

### 2a. Free wins inside the existing harness (days, $0)

- **Rebalance the time budget.** NVARC: 4B at 10h scored 27.22, at 12h scored
  29.72. More time per task clearly helps, so make sure no task is starved and
  none of the queue truncates.
- **Selection tuning.** `score_kgmon` (DFS hit count + mean augmented log-prob) is
  already the default and is the post-deadline improvement from paper §3.4. If
  Phase 1 says failures are *selection*, this is where the cheap points are —
  more re-scoring augmentations than 8, or a different aggregation.
- **More TTFT.** NVARC used 1 epoch over 128 augmented copies. If per-task time
  allows, 2 epochs or more augmentations is a one-line change.
- **DFS threshold.** `max_score = -log(0.2)` sets how many candidates get
  enumerated. Widening it trades time for recall.

Expected: a few points. Not 37.

### 2b. Retrain on the public NVARC data (weeks, ~$400–2,600 GPU rental)

This is NVARC's own thesis: the synthetic data was the innovation, and **all of
it is public**.

| Kaggle dataset | Size | Contents |
|---|---|---|
| `sorokin/nvarc-synthetic-puzzles` | 338 MB | the 103k synthetic puzzles |
| `sorokin/nvarc-augmented-puzzles` | 1.32 GB | the 3.2M-sample SFT mix |
| `sorokin/nvarc-artifacts-puzzles` | 42 GB | all intermediate generated text |

Two routes:

1. **Newer base model, same recipe.** Qwen3-4B-Thinking-2507 dates from mid-2025.
   Redo the 16-token vocabulary surgery ([cut_tokenizer.ipynb](https://github.com/1ytic/NVARC/blob/main/ARChitects/cut_tokenizer.ipynb)
   is ~10 lines of `index_select`) on a stronger 2026-era small model and SFT on
   the public 3.2M-sample mix. Plausibly the highest-leverage single change.
2. **More/better synthetic data.** Extend the SDG pipeline ([upstream SDG/](https://github.com/1ytic/NVARC/tree/main/SDG)).
   Its generator was `gpt-oss-120b`; 2026 open models are considerably better.

**Cost estimate (unverified, from public rental pricing ~$2–3/H100-hr):** NVARC's
full SFT was 32 H100 × 27h ≈ 864 H100-hours ≈ **$1,700–2,600**. A single 8×H100
node for 24h ≈ 192 H100-hours ≈ **$400–580**. The `nvarc-*` datasets carry **no
license** ("Unknown" on Kaggle) — fine to train on privately, a problem to
redistribute.

⚠️ This is the main capital decision in the plan. Do not spend it before Phase 1
says what is actually broken.

### 2c. Ensemble a second, complementary solver (weeks, $0–low)

The technique survey in Notion: induction and transduction solve *different* tasks
— ensembling reached ~56% on ARC-AGI-1. Less proven on ARC-AGI-2, but the harness
already supports injecting foreign candidates into the scoring pool (that is
exactly how NVARC bolted on TRM).

**Do not start with TRM.** NVARC's own numbers: it moved the 2B from 21.53 →
22.50 but left the 4B at 27.22 → 27.22 (paper §4.4), and cost them days of
dependency conflicts. Checkpoints are public at `cpmpml/arc-prize-trm-031` if
worth revisiting later.

---

## 6. Phase 3 — the writeup track (start now, in parallel)

**$275K goes to the best Solution Writeup and $450K to the Paper Track, and only
one of six judging criteria is your score.** For a solo entrant 37 points behind,
this is plausibly better expected value than chasing the leaderboard.

It is also cheap to run in parallel *if documentation is written as work happens*
rather than reconstructed in November. Concretely:

- Keep a decision log: what was tried, what the number was, what it implies.
  Phase 1's failure taxonomy is exactly the kind of evidence that scores on
  Theory and Completeness.
- Artifacts must be open-sourced within **7 days** of the submission deadline
  (Nov 9). Licensing: Kaggle's rules say CC BY 4.0, arcprize.org says CC0/MIT-0.
  **Dual-license your own code CC0 + MIT-0** to satisfy both.
- A negative result, cleanly measured, is publishable. "Here is precisely why
  recreating the 2025 winner yields 30% while the 2026 field reaches 67%" is a
  genuinely useful contribution.

---

## 7. Submission budget

One per day, ~88 left, but wall clock is the real constraint: a 4B run is 12
hours, so realistically **fewer than 60 useful 4B submissions** remain, and far
fewer if you sleep.

Rules of thumb:

- Never submit an uncommitted notebook. The commit run is free; the submission
  is not.
- Change **one thing** per submission, or you cannot attribute the delta.
- Expect ±1–2pp of noise (paper §3.5). A 1pp "improvement" is not an improvement.
- Two final submissions are selected for judging. Keep one safe, known-good run
  selected at all times.

---

## 8. Decision points

| When | Question | If no |
|---|---|---|
| End of Phase 0 | Does the 4B reproduce ~30%? | Harness is broken — fix before anything else. |
| End of Phase 1 | Is failure generation or selection? | Don't spend money until this is answered. |
| ~Sept 1 | Is a retrain plausibly worth $500+? | Go all-in on Phase 3 writeup instead. |
| ~Oct 1 | Is the score within reach of ~34% (8th)? | Stop optimising; spend October writing. |
| Oct 26 | Team merger deadline | Last chance to collaborate. |

**Kill criterion:** if by Oct 1 the score is not plausibly within reach of the 8th
prize position, the leaderboard is not the goal — the writeup is. Switching
deliberately beats drifting.

---

## 9. Open questions

- **Was the 2026 hidden eval refreshed from 2025?** No source states either way.
  Circumstantial evidence favours reuse (identical public files, same 240-task
  structure, described as "a relaunch"). Matters because it determines whether
  2025-era exposure to the hidden set carries over.
- **What is nvbanana doing?** Nobody publishes mid-competition. The forums have
  local-vs-LB complaints, not methods. This resolves in December, too late to
  copy — so don't wait on it.
- **Notebook RAM/disk on L4×4.** No first-party doc. Estimated ~30 GB RAM /
  ~20 GB disk. Measure inside a session before relying on it.
