# Problem 1 — `pip install` fails in the submission notebook

*2026-08-06 — first Kaggle run of the NVARC port (our recreation of the 2025 winning solution).*

> Mirrors the "First Problem" toggle in the Notion doc *Iteration 1 : Notes*.

## What broke

```text
[Errno -3] Temporary failure in name resolution
ERROR: No matching distribution found for unsloth==2025.9.7
```

## Why

The competition requires internet to be **disabled** in the submission notebook.
No network means PyPI is unreachable, so `pip install` can never work while the
notebook runs. Ours tried anyway: we copied NVARC's install steps, which were
written for their own machines, not for a Kaggle submission.

## Fix: a wheelhouse

A wheelhouse is just a folder of pre-downloaded Python packages.

1. A **separate** notebook runs with internet **on** and downloads everything we
   need. It must not have the competition attached, since attaching it forces
   internet off.
2. That notebook's saved output is attached to the submission notebook as an input.
3. The submission installs from the folder instead of the network:

```bash
pip install --no-index --find-links <wheelhouse> unsloth unsloth_zoo
```

## Four things this uncovered

**1. Kaggle's environment moved on.** NVARC built this in Dec 2025. Kaggle's
image is now about 11 months newer.

| Package | NVARC | Kaggle now |
|---|---|---|
| Python | 3.11 | 3.12 |
| torch | 2.8 | 2.10 |
| transformers | 4.55 | 5.0 |

Their pinned `unsloth==2025.9.7` would have downgraded two libraries by a whole
major version and pulled in 2.7 GB of the wrong CUDA build. Instead we install
the **current** unsloth, which fits the image, and strip torch out of the
wheelhouse so Kaggle's own version is never replaced.

**2. flash-attn is unavailable.** No prebuilt package exists for this Python and
torch combination, and compiling it takes 1–2 hours we cannot spare. Attention
falls back to SDPA, which is correct but slower. NVARC's `qwen3.patch` only
exists to wire up flash-attn, so it is dropped; the notebook detects this and
skips it instead of crashing.

**3. All three input paths were wrong.** Kaggle mounts each input type somewhere
different:

```text
/kaggle/input/<competition>/                            competition data
/kaggle/input/notebooks/<owner>/<slug>/                 notebook output
/kaggle/input/models/<owner>/<slug>/<fw>/<var>/<ver>/   models
```

We guessed these from documentation rather than checking. The code now searches
for files by name instead of assuming a location.

> **Lesson:** run a one-minute probe notebook to look at the environment before
> writing paths against it. Each wrong guess costs a full run.

**4. A dead run reported success.** The one worth remembering.

Kaggle marked a run COMPLETE even though the solver had crashed and produced
nothing:

```text
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
240 tasks, 259 outputs, 0 with a real prediction
```

A shell command that fails does **not** fail the notebook cell that ran it. So
the crash was swallowed, and we still wrote a perfectly formatted submission
full of placeholder answers. Submitting that scores **0.00** while looking like
a clean run, and you only get **one submission per day**.

Fix: capture the solver's exit code, and refuse to treat a run as good unless it
produced real predictions.

## Status

- Working: offline install, source files, data discovery, fallback submission
- Blocked: running the solver on a GPU

## Blocked on

The accelerator has to be set by hand, in the notebook editor:
**Settings → Accelerator → GPU L4 x4**. The API cannot select it. This is not
about speed. Kaggle's default P100 GPU is too old for the installed torch, which
refuses to run on it at all.

## Next unknown

The `trl` training library changed its API since NVARC's version. Their trainer
call may need updating, which is a code change rather than a settings change.
