# NVARC port — ARC Prize 2026 (ARC-AGI-2)

A recreation of the ARC Prize 2025 winning solution (Ivan Sorokin and
Jean-François Puget, NVIDIA), re-pointed at the 2026 competition. Attribution and
what we derived from where: [../NVARC/README.md](../NVARC/README.md).

**No training required** — the fine-tuned weights are public on Kaggle Models
under Apache 2.0, so this is inference only.

| Checkpoint | NVARC's 2025 public LB | Ours |
|---|---|---|
| `sorokin/qwen3_2b_grids15_sft141` | 22.22% | 20.69% |
| `sorokin/qwen3_4b_grids15_sft139` | 29.72% | **27.22%** ← current |

## Layout

```
src/arc_config.py       paths, budgets, run-mode detection      (new)
src/arc_loader.py       dataset, augmentation, submission I/O   (verbatim NVARC)
src/arc_decoder.py      candidate ranking                       (verbatim NVARC)
src/arc_mask.py         completion-only label masking           (new — see below)
src/arc_solver.py       fine-tune + batched DFS + re-scoring     (NVARC + diffs)
src/starter.py          4-GPU work queue                        (rewritten)
src/make_submission.py  aggregation + format validation         (new)
src/qwen3.patch         Unsloth batched-inference fix           (verbatim NVARC, unused)
build_notebook.py       assembles src/ into arc2_nvarc.ipynb
build_wheelhouse.py     generates the offline-package notebook
validate_local.py       28 offline checks — run before every push
analyze_candidates.py   failure taxonomy over a run's candidates
```

`src/` is the source of truth. The notebook is a build artifact; hand-edits are
overwritten by the next build.

## Workflow

```bash
python3 submission/validate_local.py      # no GPU needed
python3 submission/build_notebook.py      # --model to switch checkpoint
cd submission && kaggle kernels push
```

Then in the Kaggle editor: **Accelerator → GPU L4 ×4**, **Internet: Off**.
Commit first (4 smoke tasks), then Submit.

Inputs the notebook needs attached: the competition, the model, and the
**wheelhouse** — a notebook output holding pre-downloaded packages, because the
competition disables networking so `pip` cannot reach PyPI. Regenerate it with
`build_wheelhouse.py` only when a dependency changes.

## What the pipeline does

Per puzzle, on each of 4 GPUs pulling from a shared queue:

1. **Reset LoRA**, then fine-tune on 128 augmented copies of *that one puzzle* —
   r=256, α=32, rslora, lr 5e-5 cosine, 1 epoch, bf16. Only the answer grids are
   supervised.
2. **Batched depth-first search** over 16 augmented views (8 dihedral × 2 colour
   permutations), 4 at a time. No temperature, no fixed candidate count: it
   enumerates every completed grid whose total probability exceeds 0.2.
3. **Invert** each candidate back to canonical orientation and dedupe.
4. **Re-score** each distinct candidate under 8 augmentations — the *same* 8 for
   every candidate, which is what makes the scores comparable.
5. **Rank** by search hit count plus mean augmented log-probability, emit the top 2.

Grids are one token per cell under a 16-token vocabulary, so a 30×30 grid costs
930 tokens rather than ~6.3K.

## Changes from NVARC's original

`arc_loader.py` and `arc_decoder.py` are byte-identical. The rest:

- **Offline install** from an attached wheelhouse; internet is disabled.
- **Paths discovered by content, not shape.** Kaggle mounts competition data,
  notebook outputs and models at three different path layouts.
- **Tokeniser loaded via `PreTrainedTokenizerFast`.** `AutoTokenizer` honours the
  checkpoint's `tokenizer_class: Qwen2Tokenizer` declaration, returns the *slow*
  BPE class, and silently drops the words `user` and `assistant` — so every
  prompt was off-distribution. Worth **+7.4 points** when fixed.
- **`arc_mask.py`.** NVARC located turn boundaries by those same role token ids.
  With them missing, every label was masked and the loss was NaN over zero
  targets — the model learned nothing. Now anchored on `<|im_start|>`/`<|im_end|>`,
  which survive, so it works whether or not the role word encodes.
- **`qwen3.patch` is skipped.** No flash-attn wheel exists for this Python and
  torch, so attention falls back to SDPA. Probed and confirmed harmless: batched
  decoding matches one-at-a-time, and the search never rebinds the cache object
  it hands to sibling candidates, so they cannot interfere.
- **Gradient checkpointing on.** Numerically identical, ~20% slower, and it makes
  training memory flat at 13.1 GB regardless of puzzle size. Without flash-attn's
  savings the 4B otherwise runs close to the L4's 22 GB ceiling.
- **Failures are per-puzzle, not fatal.** One bad puzzle is skipped; previously it
  propagated through `mp.spawn` and stopped the run.
- **No wasted attempts.** Unfilled guesses were `[[0]]`, a 1×1 grid of zeros that
  is essentially never correct. They now fall back to the test input — 513
  guaranteed misses per run converted into real guesses.
- **Eval mode** (`ARC_TASK_SET=eval`) scores the 120 public evaluation puzzles in
  a commit run, costing no submission. Forced back to `test` on a scored rerun,
  since an eval-mode submission would miss every hidden id and score 0.00.
- **Durable telemetry.** One JSONL per worker with per-puzzle timing, memory and
  failures, plus exit codes captured rather than discarded.

## What to know before changing anything

**Throughput is not the constraint.** 240 puzzles take ~4–8 hours of the 12-hour
budget. Idle compute is a lever, not a limit. (Earlier versions of this file said
the opposite; it was wrong.)

**No public dataset can measure an improvement.** We score 82.6% on the public
eval split against 27.22% on the hidden set, because 55% of the checkpoint's
training data was synthesised from those puzzles' descriptions. Use eval mode as
a pre-flight check, never as a score.

**Run-to-run variance is reported at ±1–2 points** by NVARC and up to ±3.5 by
others, because batched search is nondeterministic. Anything smaller than that is
not a result. Batch orthogonal changes rather than spending a submission per
hypothesis.

**Where the remaining points are.** On the eval split, the correct grid is never
generated for 12.2% of outputs and generated-but-ranked-out for 4.1%. On failures
a median of 9 of 16 views return nothing at all, pruned by the 0.2 probability
floor — not by timeouts, and not by context limits. Re-run
`analyze_candidates.py` against any eval run to refresh this.

## Tuning knobs

Read from the environment by `arc_config.py`, all settable in notebook cell 1:

| Variable | Default | Effect |
|---|---|---|
| `ARC_MODEL_SLUG` | `qwen3_4b_grids15_sft139` | which checkpoint |
| `ARC_TASK_SET` | `test` | `eval` scores the public split for free |
| `ARC_TASK_LIMIT` | 0 | >0 samples the split for a faster signal |
| `ARC_GRAD_CKPT` | 1 | 0 is ~20% faster but risks OOM on the 4B |
| `ARC_PROBE_DFS` | 0 | 1 runs the decoder correctness probe once |
| `ARC_PUZZLE_BUDGET_S` | 1200 | per-puzzle decode ceiling |
| `ARC_DFS_BUDGET_S` | 540 | per-search-call ceiling |
| `ARC_SAFETY_MARGIN_S` | 1200 | headroom before the 12h kill |
| `ARC_NUM_WORKERS` | 4 | one per L4 |

## Deliberately not included

**TRM.** NVARC's own numbers: ensembling it moved the 2B from 21.53 → 22.50 but
left the 4B at 27.22 → 27.22, and it cost them days of dependency conflicts.
Checkpoints are public at `cpmpml/arc-prize-trm-031` if revisited — and note the
failure mode was the *selector*, not TRM.
