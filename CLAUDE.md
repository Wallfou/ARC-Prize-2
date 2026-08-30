# ARC Prize 2026 entry

**Read [HANDOFF.md](HANDOFF.md) before doing anything.** It contains the current
state, what has already been tested and rejected, and the traps. Several
obvious-looking moves have been measured and ruled out; redoing them wastes days.

Current: **29.31% / 27.64%** public leaderboard (same code, two runs).
Goal: above 30% and top 100–300. Deadline Nov 2, 2026.
Next work is **Tier A** — see HANDOFF.md §3.

## Non-negotiables

- **Noise floor is 1.67pp.** A single submission cannot resolve anything under
  ~2pp. The leaderboard confirms; it does not measure.
- **Measure offline.** `ARC_TASK_SET=eval` scores the 120 public eval tasks in a
  commit run for zero submissions, and tracks the leaderboard within ~2.4pp on
  the 4B. Prefer metrics that count events (generation recall) over ones that
  resolve a score.
- **The bottleneck is generation, not search or selection.** The correct grid is
  never produced for 67.4% of outputs; only 7.6% are generated-but-ranked-out.
  Work aimed at ranking is aimed at the small number.
- **`kaggle kernels push` resets the accelerator.** Re-set **L4 ×4** in the Kaggle
  editor after every push, or the run gets a P100, which the installed torch
  refuses to use at all.
- **The pushed notebook is currently in `eval` mode.** Flip `ARC_TASK_SET` to
  `"test"` before submitting.
- **`submission/src/` is the source of truth.** `build_notebook.py` regenerates
  the notebook; hand-edits to `arc2_nvarc.ipynb` are overwritten.
- **Run `python3 submission/validate_local.py` before every push.** 28 checks, no
  GPU needed.
- **Never score against `example/arc-agi_test_challenges.json`** — all 240 of its
  tasks are training tasks, despite what Kaggle's Data tab says.

## Watch for silent success

Four separate times a run reported COMPLETE while producing nothing useful. A
shell command's exit code does not fail a notebook cell, and a guard that infers
failure from a downstream artifact breaks when anything upstream changes that
artifact. Verify against the thing itself, and be suspicious of a green run that
did not produce the numbers you expected.
