# ARC Prize 2026 (ARC-AGI-2)

Work toward the [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
competition. Deadline **Nov 2, 2026**.

| | |
|---|---|
| [PLAN.md](PLAN.md) | **Start here.** Current iteration plan, phases, decision points. |
| [submission/](submission/README.md) | Working recreation of the 2025 winner (NVARC). Validated, ready to submit. |
| [research.md](research.md) | Technical landscape: what wins on ARC-AGI-2 and why. |
| [materials.md](materials.md) | Reading list and study sequence. |
| [NVARC/](NVARC) | Vendored source of the 2025 winning solution, including the paper. |
| [example/](example) | The competition data files. |

## Quick start

```bash
python3 submission/validate_local.py      # 20 offline checks, no GPU needed
python3 submission/build_notebook.py      # -> submission/arc2_nvarc.ipynb
```

Then upload the notebook to Kaggle, attach the competition dataset and the model
`sorokin/qwen3_2b_grids15_sft141`, set the accelerator to **L4×4** with internet
**off**, commit, and submit. Details in [submission/README.md](submission/README.md).

## One thing to know

`example/arc-agi_test_challenges.json` is **not** a sample of the evaluation set,
despite what Kaggle's Data tab says — all 240 of its tasks are *training* tasks.
Any score computed against it is meaningless. Validate on the 120-task
`arc-agi_evaluation_challenges.json` split instead.
