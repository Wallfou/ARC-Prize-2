# NVARC — reference material (not our code)

The ARC Prize 2025 winning solution by **Ivan Sorokin and Jean-François Puget**
(NVIDIA). Upstream: <https://github.com/1ytic/NVARC>.

Our submission in [../submission/](../submission) is derived from this work.
`submission/src/arc_loader.py` and `submission/src/arc_decoder.py` are
byte-identical to theirs; `arc_solver.py` is theirs with a small diff. Credit
for the method, the training pipeline and the published checkpoints is theirs.

## What is kept here, and why

| File | Why |
|---|---|
| `nvarc_2025.pdf` | The paper. Cited constantly for hyperparameters, published scores, and the things they tried and rejected. |
| `ARC-AGI1/002_ivan_arc1.ipynb` | The direct ancestor of our solver — the only public copy of the winning inference code. We diff against it whenever behaviour is in question. Outputs stripped (1.2 MB of progress bars); source intact. |

One number from those stripped outputs worth keeping: the same 4B pipeline scored
**383/400 (95.75%)** on the ARC-AGI-1 2024 public evaluation set. That appears
nowhere in the paper — which is itself a contamination signal, since 376 of the
1,000 ARC-AGI-2 training tasks *are* ARC-AGI-1 public eval tasks.

## What was removed, and where to find it

Removed on 2026-08-23 because nothing in our pipeline reads it. All of it is
still upstream, and still in this repo's git history.

| Removed | Contents | Get it from |
|---|---|---|
| `SDG/` | The synthetic-data pipeline: prompts and scripts that generated 103k puzzles | [upstream `SDG/`](https://github.com/1ytic/NVARC/tree/main/SDG) — only needed if we retrain |
| `ARChitects/` | Supervised fine-tuning config and the 16-token tokeniser surgery | [upstream `ARChitects/`](https://github.com/1ytic/NVARC/tree/main/ARChitects) — the checkpoints ship the same tokeniser files |
| `TRM/` | Tiny Recursive Model training | [upstream `TRM/`](https://github.com/1ytic/NVARC/tree/main/TRM) — dropped: their own numbers show it added nothing to the 4B (27.22 → 27.22) |
| `ARC-AGI1/pip-install-…ipynb` | Package pins and the flash-attn patch | The patch itself is preserved at `submission/src/qwen3.patch`; the pins are in `submission/build_wheelhouse.py` |
| `external/` | Empty submodule directories | Never populated |

## Published artifacts we actually use

- `sorokin/qwen3_4b_grids15_sft139` — 7.27 GB, Apache 2.0. The checkpoint we run.
- `sorokin/qwen3_2b_grids15_sft141` — 2.82 GB, Apache 2.0. Smaller, faster variant.
- `sorokin/nvarc-synthetic-puzzles`, `sorokin/nvarc-augmented-puzzles` — training data, **no licence stated**. Fine to train on privately; a problem to redistribute.
