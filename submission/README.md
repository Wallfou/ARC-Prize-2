# NVARC port — ARC Prize 2026 (ARC-AGI-2)

A recreation of the ARC Prize 2025 winning solution (Ivan Sorokin and
Jean-François Puget, NVIDIA), re-pointed at the 2026 competition.

**No training required.** The fine-tuned weights are public on Kaggle Models
under Apache 2.0, so this is inference only. See [../NVARC/nvarc_2025.pdf](../NVARC/nvarc_2025.pdf)
for the method and [../research.md](../research.md) for the wider landscape.

| Model | Runtime | Public LB (2025) |
|---|---|---|
| `sorokin/qwen3_2b_grids15_sft141` | 6h10m | 22.22% |
| `sorokin/qwen3_4b_grids15_sft139` | 12h | 29.72% |

Start with the 2B. It finishes in half the window, so a timeout or OOM costs
one submission day instead of silently truncating the task queue at hour 12.

## Layout

```
src/arc_config.py       paths, budgets, run-mode detection      (new)
src/arc_loader.py       dataset, augmentation, submission I/O   (verbatim NVARC)
src/arc_decoder.py      candidate ranking                       (verbatim NVARC)
src/arc_solver.py       TTFT + batched DFS + re-scoring         (NVARC, 6-line diff)
src/starter.py          4-GPU work queue                        (rewritten for Kaggle)
src/make_submission.py  aggregation + format validation         (new)
src/qwen3.patch         Unsloth batched-inference fix           (verbatim NVARC)
build_notebook.py       assembles src/ into arc2_nvarc.ipynb
validate_local.py       offline checks — run before every commit
```

`src/` is the source of truth; the notebook is a build artifact.

## Workflow

```bash
python3 submission/validate_local.py && python3 submission/build_notebook.py
```

Then on Kaggle: upload `submission/arc2_nvarc.ipynb`, attach the competition
dataset and the model, set accelerator **L4x4**, internet **off**, and submit.

For the 4B run:

```bash
python3 submission/build_notebook.py --model qwen3_4b_grids15_sft139
```

## What the pipeline does

Per task, on each of 4 GPUs pulling from a shared queue:

1. **Reset LoRA**, then test-time fine-tune on 128 augmented copies of *that one
   task* — r=256, α=32, rslora, lr 5e-5 cosine, 1 epoch, bf16.
2. **Batched DFS decode** over 16 augmented views (8 dihedral × 2 colour
   permutations), batched 4 at a time. No temperature and no fixed candidate
   count: it enumerates every completion whose total probability exceeds 0.2
   (`max_score = -log(0.2)`).
3. **Invert** each candidate back to canonical orientation and dedupe.
4. **Re-score** each unique candidate under 8 augmentations — the *same* 8 for
   every candidate of a task, which is what makes the scores comparable.
5. **Rank** by `score_kgmon` = DFS hit count + mean augmented log-probability,
   and emit the top 2 distinct grids.

Grids are encoded one token per cell under a 16-token vocabulary (digits 0–9,
newline, `user`, `assistant`, and 3 specials), which is why a 30×30 grid costs
930 tokens instead of ~6.3K.

## Changes from NVARC's original

`arc_loader.py` and `arc_decoder.py` are byte-identical. Everything else:

- **Data paths** → `/kaggle/input/arc-prize-2026-arc-agi-2/`, reading
  `arc-agi_test_challenges.json`.
- **Writable paths.** NVARC wrote to `../inference_outputs` and `../worker{rank}`,
  which resolve to `/kaggle/` on Kaggle — not writable. Moved under
  `/kaggle/working/`.
- **Model path** is globbed, not hardcoded: Kaggle mounts models at
  `<slug>/<framework>/<variation>/<version>/` and the version changes per attach.
- **Rerun mode restored.** The original had `rerun_mode = True` hardcoded in
  `arc_solver.py`, `False` in the decoder cell, and both branches pointing at the
  same file. Now reads `KAGGLE_IS_COMPETITION_RERUN`.
- **Fallback submission written first**, before any GPU work, so a crash or an
  overrun still leaves a scoreable file.
- **Format validation** that raises rather than submitting something Kaggle
  scores as zero.
- **Unsloth patch target discovered dynamically** instead of the hardcoded
  `/usr/local/lib/python3.11/dist-packages`, and asserted afterwards.
- **Pace instrumentation** — each worker logs its running average and how many
  more tasks fit before the deadline.

## Risks

**The Unsloth pin is the fragile part.** `qwen3.patch` rewrites Unsloth's
inference path to call `flash_attn_func`, which is what makes batched decoding
(the "batch4" in NVARC's notebook name) work at all. Its hunks are line-matched
to `unsloth==2025.9.7` — a December 2025 release being installed into an August
2026 Kaggle image. If it fails to apply, the notebook raises immediately rather
than running unpatched. Expect this to be the first thing that breaks.

**Throughput, not accuracy, is the binding constraint.** 240 tasks ÷ 4 workers ×
the 1200s per-task ceiling = 20 hours against a 12-hour wall. The ceiling is a
ceiling, not a target — NVARC's average came in well under it — but if the pace
log shows otherwise, lower `ARC_PUZZLE_BUDGET_S` or `ARC_DFS_BUDGET_S` rather
than letting the queue truncate.

**Never validate on `arc-agi_test_challenges.json`.** Kaggle's Data tab says it
is a placeholder drawn from the evaluation set. It is not: all 240 tasks are
*training* tasks, byte-identical demo pairs included. `validate_local.py`
asserts this. Score on the 120-task public eval split instead.

**Even the public eval split reads high.** NVARC seeded 55% of their training
data from ARC-AGI-2 public *eval* descriptions — they hand-labelled 29 of those
puzzles and LLM-generated summaries for the other 91 (paper §2.1, §3.5). The
released checkpoints have semantic exposure to exactly those 120 tasks.

**Run-to-run variance is 1–2 points** on identical notebooks (paper §3.5), since
batched DFS is nondeterministic. NVARC tried Thinking Machines' `batch_invariant_ops`
to fix it; 17% slower, so they dropped it.

## Deliberately not included

**TRM.** NVARC's own numbers: ensembling it moved the 2B from 21.53 → 22.50, but
left the 4B at 27.22 → 27.22 (paper §4.4). Most TRM-solved puzzles were already
solved by Qwen3. It also cost them days of dependency conflicts. Checkpoints are
public at `cpmpml/arc-prize-trm-031` if you want to revisit it.

## Tuning knobs

All read from the environment by `arc_config.py`:

| Variable | Default | Effect |
|---|---|---|
| `ARC_MODEL_SLUG` | `qwen3_2b_grids15_sft141` | which checkpoint |
| `ARC_PUZZLE_BUDGET_S` | 1200 | per-task decode ceiling |
| `ARC_DFS_BUDGET_S` | 540 | per-DFS-call ceiling |
| `ARC_SAFETY_MARGIN_S` | 1200 | headroom before the 12h kill |
| `ARC_NUM_WORKERS` | 4 | one per L4 |
