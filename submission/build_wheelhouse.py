"""Generate a Kaggle notebook that builds an offline wheelhouse.

The competition forbids internet access in the submission notebook, so
`pip install unsloth` cannot run there. This builder is a *separate* notebook
with internet ON (it must NOT be attached to the competition, or Kaggle forces
internet off). It downloads every wheel the solver needs into /kaggle/working,
and its saved output is then attached to the submission notebook.

    python3 submission/build_wheelhouse.py     # writes wheelhouse/
    cd submission/wheelhouse && kaggle kernels push

Then check the log, and attach `wallfou/arc2-wheelhouse` to the submission
notebook as a notebook-output input.
"""

import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

# Pins from NVARC's pip-install-unsloth-flash-patch.ipynb. qwen3.patch is
# line-matched to unsloth 2025.9.7, so that pin is load-bearing.
PINS = ["unsloth==2025.9.7", "unsloth_zoo==2025.9.9"]

REPORT = '''\
import sys, subprocess, importlib.metadata as md

print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
          "| cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
except Exception as e:
    print("torch import failed:", e)

# What does the image already ship? Anything present here need not be downloaded.
for p in ["unsloth", "unsloth_zoo", "flash_attn", "transformers", "peft", "trl",
          "datasets", "accelerate", "bitsandbytes", "triton", "xformers"]:
    try:
        print(f"  {p:16s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:16s} -")
'''

DOWNLOAD = '''\
import os, subprocess, sys, glob, shutil

# The image's torch is CUDA-matched to its driver. Never ship a second one: a
# generic torch wheel drags ~2.7GB of CUDA-13 libs and would shadow it.
EXCLUDE = ("torch-", "torchvision-", "torchaudio-", "triton-", "nvidia_",
           "cuda_bindings", "cuda_pathfinder")

def strip_torch(d):
    dropped = 0
    for f in glob.glob(os.path.join(d, "*")):
        if os.path.basename(f).startswith(EXCLUDE):
            os.remove(f); dropped += 1
    print(f"  dropped {dropped} torch/CUDA wheels from {os.path.basename(d)}")

def size(d):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*"))) / 1e6

# --- Variant A: NVARC's exact pins, no deps -------------------------------
# unsloth 2025.9.7 requires transformers 4.55.4 / datasets 3.6.0, i.e. a
# two-major-version downgrade of what this image ships. Grab the packages
# themselves without deps so the pinned path stays testable, cheaply.
PINNED = "/kaggle/working/wheels_pinned"
os.makedirs(PINNED, exist_ok=True)
r = subprocess.run([sys.executable, "-m", "pip", "download", "--dest", PINNED,
                    "--no-deps", "--only-binary", ":all:", *__PINS__],
                   capture_output=True, text=True)
print("A (pinned, --no-deps) exit:", r.returncode)
print((r.stdout or r.stderr)[-1500:])

# --- Variant B: current unsloth, full deps, torch excluded -----------------
# Most likely to work against py3.12 / torch 2.10 / transformers 5.
LATEST = "/kaggle/working/wheels"
os.makedirs(LATEST, exist_ok=True)
r = subprocess.run([sys.executable, "-m", "pip", "download", "--dest", LATEST,
                    "--only-binary", ":all:", "unsloth", "unsloth_zoo"],
                   capture_output=True, text=True)
print("\\nB (latest, full deps) exit:", r.returncode)
print((r.stdout or r.stderr)[-2500:])

for d in (PINNED, LATEST):
    strip_torch(d)
    print(f"  {os.path.basename(d)}: {size(d):.0f} MB")

# What version did "latest" actually resolve to, and what does it want?
for f in sorted(glob.glob(os.path.join(LATEST, "unsloth*"))):
    print("  ->", os.path.basename(f))
for name in ("transformers", "datasets", "peft", "trl", "tokenizers"):
    for f in sorted(glob.glob(os.path.join(LATEST, f"{name}-*"))):
        print("  ->", os.path.basename(f))
'''

FLASH = '''\
# flash-attn ships no generic PyPI wheel -- building from source takes 1-2h,
# which would eat the submission budget. Fetch a prebuilt wheel from the
# Dao-AILab releases matching this exact python / torch / CUDA / ABI combo.
import sys, torch, subprocess, os, urllib.request

WHEELS = "/kaggle/working/wheels"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
tv = ".".join(torch.__version__.split(".")[:2])
cu = "cu12" if (torch.version.cuda or "12").startswith("12") else "cu11"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(f"target: {py} torch{tv} {cu} cxx11abi{abi}")

# Ask the GitHub API which assets actually exist rather than guessing tags.
import json as _json
try:
    req = urllib.request.Request(
        "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=15",
        headers={"Accept": "application/vnd.github+json"})
    releases = _json.load(urllib.request.urlopen(req, timeout=60))
except Exception as e:
    releases = []
    print("GitHub API unreachable:", e)

want = f"torch{tv}"
matches = []
for rel in releases:
    for asset in rel.get("assets", []):
        n = asset["name"]
        if n.endswith(".whl") and py in n and want in n and cu in n and f"abi{abi}" in n:
            matches.append((rel["tag_name"], n, asset["browser_download_url"]))

print(f"assets matching {py}/{want}/{cu}/abi{abi}: {len(matches)}")
got = None
for tag, name, url in matches[:3]:
    dest = os.path.join(WHEELS, name)
    try:
        urllib.request.urlretrieve(url, dest)
        mb = os.path.getsize(dest) / 1e6
        if mb < 1:
            os.remove(dest); raise ValueError("not a wheel")
        print(f"OK {name} ({mb:.0f} MB)")
        got = name
        break
    except Exception as e:
        print(f"  failed {tag}: {type(e).__name__}")

if not got:
    # Show what IS available for this python, so the mismatch is obvious.
    avail = sorted({a["name"] for r in releases for a in r.get("assets", [])
                    if a["name"].endswith(".whl") and py in a["name"]})
    print(f"\\nNo flash-attn wheel for torch {tv}. Available for {py}:")
    for n in avail[:12]:
        print("   ", n)
    print("\\nThe solver falls back to SDPA, which is correct but slower.")
    print("qwen3.patch is skipped automatically when flash_attn is absent.")
'''

MANIFEST = '''\
import os, glob, json, subprocess, sys

WHEELS = "/kaggle/working/wheels"

for d in ("/kaggle/working/wheels", "/kaggle/working/wheels_pinned"):
    files = sorted(glob.glob(os.path.join(d, "*")))
    total = sum(os.path.getsize(f) for f in files)
    print(f"{os.path.basename(d)}: {len(files)} files, {total/1e6:.0f} MB")
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump([{"name": os.path.basename(f), "bytes": os.path.getsize(f)}
                   for f in files], fh, indent=2)

# Prove variant B resolves offline WITHOUT pulling torch. --dry-run reports what
# it would do; anything mentioning torch means the exclusion leaked.
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--dry-run", "--no-index",
     "--find-links", WHEELS, "unsloth", "unsloth_zoo"],
    capture_output=True, text=True)
out = r.stdout or r.stderr
print("\\n--- variant B offline dry-run ---")
for line in out.splitlines():
    if line.startswith("Would install") or "ERROR" in line:
        print(line[:2000])
print("exit:", r.returncode)

if r.returncode == 0:
    would = [l for l in out.splitlines() if l.startswith("Would install")]
    if would and "torch-" in would[0]:
        print("\\nWARNING: resolution still wants torch -- do not install this blindly")
    print("\\nOK: variant B resolves offline")
else:
    print("\\nvariant B does NOT resolve. Inspect the ERROR above.")
'''


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def build(out_dir, username):
    os.makedirs(out_dir, exist_ok=True)

    cells = [
        markdown(
            "# ARC 2026 wheelhouse\n"
            "\n"
            "Downloads the packages the NVARC solver needs, so the submission\n"
            "notebook can install them with internet **off**.\n"
            "\n"
            "**This notebook needs internet ON and must NOT have the competition\n"
            "attached** -- attaching it forces internet off.\n"
            "\n"
            "Generated by `submission/build_wheelhouse.py`.\n"
        ),
        code(REPORT),
        code(DOWNLOAD.replace("__PINS__", repr(PINS))),
        code(FLASH),
        code(MANIFEST),
    ]

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

    nb_name = "arc2_wheelhouse.ipynb"
    with open(os.path.join(out_dir, nb_name), "w") as f:
        json.dump(nb, f, indent=1)

    meta = {
        "id": f"{username}/arc2-wheelhouse",
        "title": "arc2-wheelhouse",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # GPU needed: flash-attn wheel selection reads torch.version.cuda and
        # the CXX11 ABI flag, which must match the machine the solver runs on.
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out_dir}/{nb_name}")
    print(f"wrote {out_dir}/kernel-metadata.json  (id: {meta['id']})")
    print(f"\nnext:  cd {out_dir} && kaggle kernels push")


if __name__ == "__main__":
    import build_notebook

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "wheelhouse"))
    p.add_argument("--username", default=None)
    a = p.parse_args()
    build(a.out, a.username or build_notebook.kaggle_username() or "YOUR_KAGGLE_USERNAME")
