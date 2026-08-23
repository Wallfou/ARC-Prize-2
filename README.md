# ARC Prize 2026 (ARC-AGI-2)

Competition entry for [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).
Final submission deadline **Nov 2, 2026**.

**Current: 29.31% public leaderboard** (v34, 4B checkpoint). Progression
13.33% → 20.69% → 27.22% → 29.31%, matching NVARC's published 29.72% for this
checkpoint. Rank ~602 of 1594; the median working submission is 29.86.

## Layout

| | |
|---|---|
| [PLAN.md](PLAN.md) | **Start here.** Status, corrections to earlier assumptions, next levers, decision points. |
| [submission/](submission/README.md) | The entry: a port of the 2025 winner. How it works, how to build and run it. |
| [NVARC/](NVARC/README.md) | Reference material from the 2025 winner. Not our code — see it for attribution. |
| [example/](example) | The competition data files, needed by the offline tests. |

Narrative history, per-iteration debugging notes, research, and background
reading live in Notion under **Projects → ARC-Prize**, not in this repo:

- *Iteration 1 : Notes* — the three silent-failure bugs that cost the first three submissions
- *Iteration 2 : Notes* — current work: diagnosis, Tier 0 fixes, the decoder probe
- *Research for Next Steps* — tiered list of levers, with what is verified and what is not
- *Background Reading & Technical Landscape* — technique survey and reading list

## Quick start

```bash
python3 submission/validate_local.py      # 28 offline checks, no GPU needed
python3 submission/build_notebook.py      # -> submission/arc2_nvarc.ipynb
cd submission && kaggle kernels push      # -> wallfou/arc2-nvarc-2026
```

Then in the Kaggle editor set **Accelerator → GPU L4 ×4** and confirm
**Internet: Off**. Commit first (4 smoke tasks, minutes), then submit.

## Four things that will bite you

**Pushing resets the accelerator.** Every `kaggle kernels push` reverts the
notebook to the default GPU, which is a P100 — and the image's torch 2.10
refuses to run on it at all (`sm_60` below its `sm_70` minimum). Re-set L4 ×4
after every push, before running.

**`example/arc-agi_test_challenges.json` is not a sample of the eval set**,
despite what Kaggle's Data tab says. All 240 of its tasks are *training* tasks.
Any score computed against it is meaningless. `validate_local.py` asserts this.

**The public eval split cannot measure improvement.** We score 82.6% on it
against 27.22% on the hidden set, because 55% of the checkpoint's training data
was synthesised from those very puzzles' descriptions. Use it as a pre-flight
check that the pipeline runs, never as a score. The leaderboard is the only
honest signal, at one submission per day.

**Silent success is the recurring failure mode.** Three times now, something
broke while the notebook reported COMPLETE and wrote a well-formed submission
that would have scored ~0. A shell command's exit code does not fail a notebook
cell; a guard that infers failure from a downstream artifact breaks when
anything upstream changes that artifact. Verify against the thing itself.

## Editing

`submission/src/` is the source of truth. `build_notebook.py` generates the
notebook from it — never hand-edit `arc2_nvarc.ipynb`, the next build overwrites
it. Run `validate_local.py` before every push.
