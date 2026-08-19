"""Assemble submission/src/*.py into a self-contained Kaggle notebook.

The repo stays the source of truth; the notebook is a build artifact:

    python3 submission/build_notebook.py

Then upload submission/arc2_nvarc.ipynb to Kaggle, attach the competition
dataset and the model, and set the accelerator to L4x4 with internet OFF.
"""

import os
import json
import base64
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

# Order matters: arc_config first, arc_solver imports arc_loader.
PY_FILES = [
    "arc_config.py",
    "arc_loader.py",
    "arc_mask.py",
    "arc_decoder.py",
    "arc_solver.py",
    "make_submission.py",
    "starter.py",
]

# Offline install. The competition requires internet OFF, so pip cannot reach
# PyPI -- everything comes from the wheelhouse notebook's saved output. Build it
# with `python3 submission/build_wheelhouse.py` and attach its output here.
#
# NVARC pinned unsloth==2025.9.7, but that release wants transformers 4.55 /
# datasets 3.6 while this image ships 5.x. The current unsloth resolves against
# the image cleanly, so we install unpinned and skip qwen3.patch (see below).
INSTALL = '''\
import glob, os, subprocess, sys

# Locate the wheelhouse by content, not by path. Datasets mount at
# /kaggle/input/<slug>/ but notebook outputs land at
# /kaggle/input/notebooks/<owner>/<slug>/, so hunt for the wheels themselves.
found = glob.glob("/kaggle/input/**/unsloth-*.whl", recursive=True)
# Prefer the full dependency set over the --no-deps 'wheels_pinned' variant.
dirs = sorted({os.path.dirname(p) for p in found},
              key=lambda d: (d.endswith("wheels_pinned"), -len(os.listdir(d))))
assert dirs, (
    "No wheelhouse found under /kaggle/input. Internet is off in this "
    "competition, so pip cannot reach PyPI. Run submission/build_wheelhouse.py, "
    "push and run that notebook, then attach its output here via "
    "Add Input -> Notebook Output.\\n"
    "Present: " + str(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"))
)
WHEELS = dirs[0]
print("wheelhouse:", WHEELS, "->", len(os.listdir(WHEELS)), "files")

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"],
               capture_output=True)

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", WHEELS,
     "unsloth", "unsloth_zoo"],
    capture_output=True, text=True)
print(r.stdout[-3000:] or r.stderr[-3000:])
assert r.returncode == 0, "unsloth install failed -- see log above"

# flash-attn is optional: without it Unsloth falls back to SDPA, which is slower
# but correct. Only install if a matching prebuilt wheel is in the wheelhouse.
fa = glob.glob(os.path.join(WHEELS, "flash_attn-*.whl"))
if fa:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "--no-deps", fa[0]], capture_output=True, text=True)
    print("flash-attn:", "OK" if r.returncode == 0 else r.stderr[-1500:])
else:
    print("WARNING: no flash-attn wheel; falling back to SDPA (slower)")
    os.environ["NO_FLASH_ATTN"] = "1"'''

# NVARC hardcoded /usr/local/lib/python3.11/dist-packages. Locate it instead:
# the Kaggle image has moved since Dec 2025. Fail loudly if the patch misses --
# unpatched Unsloth silently mishandles batched inference, and the whole DFS
# runs with batch size 4.
PATCH_APPLY = '''\
import base64, os, subprocess, sys

# The patch rewrites Unsloth's inference path to call flash_attn_func, so it is
# only valid when flash-attn is installed. Without it, leave Unsloth on SDPA:
# batched decoding still works, just slower.
try:
    import flash_attn
    HAVE_FA = True
    print("flash_attn", flash_attn.__version__)
except ImportError:
    HAVE_FA = False
    print("flash_attn absent -- skipping patch, Unsloth will use SDPA")

if HAVE_FA:
    with open("qwen3.patch", "wb") as f:
        f.write(base64.b64decode(PATCH_B64))

    import unsloth.models
    target = os.path.join(os.path.dirname(unsloth.models.__file__), "qwen3.py")
    print("patch target:", target)

    r = subprocess.run(["patch", "--binary", "--forward", target, "qwen3.patch"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)

    src = open(target).read()
    assert "flash_attn_func(Qnn, Knn, Vnn)" in src, (
        "qwen3.py is NOT patched. Batched DFS decoding wants Unsloth's inference "
        "path on flash_attn_func with bsz>1. Check that unsloth==2025.9.7 "
        "installed and that the patch hunks still apply to this version."
    )
    print("OK: unsloth qwen3 inference path patched for batched decoding")'''


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def build(model_slug, out_path):
    with open(os.path.join(SRC, "qwen3.patch"), "rb") as f:
        patch_b64 = base64.b64encode(f.read()).decode()

    cells = [
        markdown(
            "# NVARC port -- ARC Prize 2026 (ARC-AGI-2)\n"
            "\n"
            "Recreation of the ARC Prize 2025 winning solution by Ivan Sorokin and\n"
            "Jean-Francois Puget (NVIDIA), re-pointed at the 2026 competition.\n"
            "Generated by `submission/build_notebook.py` -- edit the sources in\n"
            "`submission/src/`, not this notebook.\n"
            "\n"
            "**Setup:** accelerator `L4x4`, internet **off**, attach the competition\n"
            f"dataset and the Kaggle model `sorokin/{model_slug}`.\n"
        ),
        code(
            "import time, os\n"
            "NOTEBOOK_START = time.time()\n"
            f'os.environ["ARC_MODEL_SLUG"] = "{model_slug}"\n'
            "\n"
            "# 'test'  -> the scored path (240 hidden tasks on a rerun, 4 smoke\n"
            "#            tasks on a commit).\n"
            "# 'eval'  -> the 120 public evaluation tasks, whose answers ship with\n"
            "#            the competition. A commit run then prints a real accuracy\n"
            "#            number and costs zero submissions. NVARC's published\n"
            "#            baseline on this split is 25/120 for the 2B, 30/120 for 4B.\n"
            'os.environ["ARC_TASK_SET"] = "eval"   # <- "test" before submitting\n'
            'os.environ["ARC_TASK_LIMIT"] = "0"   # >0 samples the split\n'
            'print("start", time.strftime("%H:%M:%S"), "| set",'
            ' os.environ["ARC_TASK_SET"])'
        ),
        code(INSTALL),
        code(f'PATCH_B64 = "{patch_b64}"'),
        code(PATCH_APPLY),
    ]

    for name in PY_FILES:
        with open(os.path.join(SRC, name)) as f:
            body = f.read()
        cells.append(code(f"%%writefile {name}\n{body}"))

    cells.append(code(
        "# Write a format-valid submission before any GPU work, so a crash or a\n"
        "# 12h timeout still leaves a scoreable file on disk.\n"
        "import sys; sys.path.insert(0, '.')\n"
        "import arc_config\n"
        "from make_submission import write_fallback\n"
        "write_fallback()"
    ))

    cells.append(code(
        "# Budget is anchored at notebook start, so pip install time counts.\n"
        "end_time = NOTEBOOK_START + arc_config.TOTAL_BUDGET_S - arc_config.SAFETY_MARGIN_S\n"
        'print(f"solver budget: {(end_time - time.time()) / 3600:.2f}h")'
    ))

    cells.append(code(
        "# A bare `!python` cannot fail a papermill cell, so a crashed solver would\n"
        "# look like a green run that quietly submits all-[[0]]. Capture the code.\n"
        "import subprocess, os\n"
        "env = dict(os.environ, UNSLOTH_DISABLE_STATISTICS='1',\n"
        "           TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',\n"
        "           OMP_NUM_THREADS='12')\n"
        "proc = subprocess.run(['python', 'starter.py', '--end-time', str(end_time)],\n"
        "                      env=env)\n"
        "SOLVER_RC = proc.returncode\n"
        "print('solver exit code:', SOLVER_RC)"
    ))

    cells.append(code(
        "# Rank candidates and overwrite the fallback. Raises if the format is bad.\n"
        "!python make_submission.py --selector kgmon\n"
        'print(f"total elapsed: {(time.time() - NOTEBOOK_START) / 3600:.2f}h")'
    ))

    cells.append(code(
        "import json\n"
        "sub = json.load(open(arc_config.SUBMISSION_PATH))\n"
        "n_out = sum(len(v) for v in sub.values())\n"
        "solved = sum(1 for v in sub.values()\n"
        "             for a in v if a['attempt_1'] != [[0]])\n"
        "print(f'{len(sub)} tasks, {n_out} outputs, {solved} with a real prediction')\n"
        "print('solver exit code:', SOLVER_RC)\n"
        "\n"
        "if solved == 0:\n"
        "    msg = ('ZERO real predictions -- the solver produced nothing. This '\n"
        "           'submission would score 0.00. Check the solver traceback above; '\n"
        "           'the usual cause is no GPU (needs L4x4) or a bad model path.')\n"
        "    if arc_config.is_rerun():\n"
        "        # Scored rerun: keep the format-valid fallback rather than erroring\n"
        "        # out with no submission at all, but make the failure unmissable.\n"
        "        print('!!! ' + msg)\n"
        "    else:\n"
        "        # Commit run: fail loudly so this never gets promoted to a submission.\n"
        "        raise AssertionError(msg)\n"
        "else:\n"
        "    print('OK: solver produced real predictions')"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out_path} ({len(cells)} cells, model={model_slug})")

    write_kernel_metadata(model_slug, out_path)


def kaggle_username():
    """Find the authenticated Kaggle username.

    Two auth methods exist and store credentials differently: the older API
    token writes ~/.kaggle/kaggle.json, while `kaggle auth login` (OAuth)
    writes ~/.kaggle/access_token with the username only in kaggle.config.
    Try the env var, then the JSON token, then ask the CLI.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    cred = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(cred):
        try:
            with open(cred) as f:
                name = json.load(f).get("username")
            if name:
                return name
        except (ValueError, OSError):
            pass

    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True,
                             text=True, timeout=30).stdout
        for line in out.splitlines():
            if "username:" in line:
                name = line.split("username:", 1)[1].strip()
                if name and name != "None":
                    return name
    except (OSError, subprocess.SubprocessError):
        pass

    return None


def write_kernel_metadata(model_slug, notebook_path, version="1"):
    """Emit kernel-metadata.json so the notebook can be pushed by CLI.

    Note: Kaggle's metadata schema has no field for *which* accelerator, so
    L4x4 still has to be selected once in the notebook editor. After that,
    `kaggle kernels push` preserves it.
    """
    username = kaggle_username() or "YOUR_KAGGLE_USERNAME"

    meta = {
        "id": f"{username}/arc2-nvarc-2026",
        "title": "arc2-nvarc-2026",
        "code_file": os.path.basename(notebook_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        # The wheelhouse notebook's output supplies the offline packages.
        "kernel_sources": [f"{username}/arc2-wheelhouse"],
        "model_sources": [f"sorokin/{model_slug}/transformers/bfloat16/{version}"],
        # NOTE: deliberately no "machine_shape". Kaggle normalises any value we
        # send to "Gpu" (= the default P100), which would silently CLOBBER an
        # L4x4 accelerator set in the UI. Omitting it leaves the UI choice alone.
        # torch 2.10 needs sm_70+, and P100 is sm_60, so L4x4 is mandatory.
    }
    path = os.path.join(os.path.dirname(notebook_path), "kernel-metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    note = "" if username != "YOUR_KAGGLE_USERNAME" else "  <- edit the username"
    print(f"wrote {path}{note}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3_2b_grids15_sft141",
                   help="qwen3_2b_grids15_sft141 (6h, ~22%%) "
                        "or qwen3_4b_grids15_sft139 (12h, ~30%%)")
    p.add_argument("--out", default=os.path.join(HERE, "arc2_nvarc.ipynb"))
    a = p.parse_args()
    build(a.model, a.out)
